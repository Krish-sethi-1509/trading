"""Integration smoke checks for the Step 3 ML, Step 6 API, and Step 8 config.

Run from any working directory:
    python ml/test_pipeline.py

The script performs a read-only `SELECT 1` against DATABASE_URL, loads the
saved model artifact, calculates features from synthetic bars, and then runs
FastAPI endpoint checks against an isolated temporary SQLite database. Network
price/LLM calls are mocked; API key *presence* is checked without displaying
secret values. Missing model/provider setup is reported as a failure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
ML_DIR = ROOT / "step3"
sys.path.insert(0, str(BACKEND_DIR))

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    state = "PASS" if passed else "FAIL"
    print(f"[{state}] {name}" + (f" — {detail}" if detail else ""))


def check_key_presence() -> None:
    price_provider_set = bool(os.getenv("GOLD_API_KEY") or os.getenv("GOLDAPI_API_KEY") or os.getenv("TWELVE_DATA_API_KEY"))
    report("live-price API key", price_provider_set, "GoldAPI or Twelve Data key configured" if price_provider_set else "set GOLDAPI_API_KEY or TWELVE_DATA_API_KEY")
    report("Twelve Data ingestion key", bool(os.getenv("TWELVE_DATA_API_KEY")), "configured" if os.getenv("TWELVE_DATA_API_KEY") else "set TWELVE_DATA_API_KEY for historical market ingestion")
    report("FRED macro key", bool(os.getenv("FRED_API_KEY")), "configured" if os.getenv("FRED_API_KEY") else "set FRED_API_KEY for DFII10 history")

    search_provider = os.getenv("SEARCH_PROVIDER", "tavily").strip().lower()
    search_key = "TAVILY_API_KEY" if search_provider == "tavily" else "SERPER_API_KEY" if search_provider == "serper" else None
    report(
        "RAG search API key",
        bool((search_key and os.getenv(search_key)) or os.getenv("SEARCH_API_KEY")),
        f"{search_key or 'SEARCH_API_KEY'} configured" if (search_key and os.getenv(search_key)) or os.getenv("SEARCH_API_KEY") else f"configure a supported SEARCH_PROVIDER and its API key (selected: {search_provider})",
    )
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    llm_key = "OPENAI_API_KEY" if llm_provider == "openai" else "ANTHROPIC_API_KEY" if llm_provider == "anthropic" else None
    report(
        "LLM API key",
        bool((llm_key and os.getenv(llm_key)) or os.getenv("LLM_API_KEY")),
        f"{llm_key or 'LLM_API_KEY'} configured" if (llm_key and os.getenv(llm_key)) or os.getenv("LLM_API_KEY") else f"configure a supported LLM_PROVIDER and its API key (selected: {llm_provider})",
    )


def check_live_database(database_url: str | None) -> None:
    if not database_url:
        report("configured database connection", False, "DATABASE_URL is not set")
        return
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url, pool_pre_ping=True, pool_timeout=5)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1")).scalar_one()
        finally:
            engine.dispose()
        report("configured database connection", True, "read-only SELECT 1 succeeded")
    except Exception as exc:
        report("configured database connection", False, f"{type(exc).__name__}: {exc}")


def check_feature_pipeline() -> None:
    module_path = ML_DIR / "feature_engineering.py"
    spec = importlib.util.spec_from_file_location("gold_feature_engineering", module_path)
    if spec is None or spec.loader is None:
        report("feature calculation pipeline", False, f"cannot load {module_path}")
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        import numpy as np
        import pandas as pd

        timestamps = pd.date_range("2025-01-06T00:00:00Z", periods=864, freq="5min")
        index = np.arange(len(timestamps), dtype=float)
        close = 2600.0 + np.sin(index / 7.0) * 3.0 + index * 0.002
        gold = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close - 0.25,
                "high": close + 1.0 + (index % 3) * 0.1,
                "low": close - 1.0 - (index % 2) * 0.1,
                "close": close,
                "volume": 100.0 + (index % 11) * 10.0,
            }
        )
        with tempfile.TemporaryDirectory(prefix="gold-feature-check-") as directory:
            base = timestamps[0].normalize() - pd.Timedelta(days=1)
            dates = pd.date_range(base, periods=6, freq="D", tz="UTC")
            tips_path = Path(directory) / "tips.csv"
            dxy_path = Path(directory) / "dxy.csv"
            pd.DataFrame({"timestamp": dates, "tips_yield": [1.8, 1.79, 1.77, 1.75, 1.76, 1.74]}).to_csv(tips_path, index=False)
            pd.DataFrame({"timestamp": dates, "dxy_close": [104.0, 103.8, 103.6, 103.9, 103.7, 103.5]}).to_csv(dxy_path, index=False)
            result = module.build_features(gold, tips=str(tips_path), dxy=str(dxy_path))
        expected = {
            "overlap_vwap", "overlap_vwap_plus_2sigma", "liquidity_high_sweep",
            "liquidity_low_sweep", "relative_volume", "real_yield_divergence",
            "real_yield_gold_lag_signal", "dxy_return", "atr_14", "bullish_fvg",
        }
        missing = expected - set(result.columns)
        overlap_values = result.loc[result["is_london_ny_overlap"].eq(1), "overlap_vwap"].notna().sum()
        if missing or overlap_values == 0:
            report("feature calculation pipeline", False, f"missing={sorted(missing)}, overlap VWAP rows={overlap_values}")
        else:
            report("feature calculation pipeline", True, f"{len(result)} synthetic bars; institutional feature columns and overlap VWAP populated")
    except Exception as exc:
        report("feature calculation pipeline", False, f"{type(exc).__name__}: {exc}")


def check_model_artifact() -> None:
    try:
        import services

        model, scaler, features = services._load_artifacts()
        valid = callable(getattr(model, "predict_proba", None))
        if not valid:
            raise TypeError("loaded model does not implement predict_proba")
        if scaler is None and not hasattr(model, "named_steps"):
            raise TypeError("expected fitted pipeline or a separate SCALER_PATH")
        if not features:
            raise TypeError("model feature order unavailable; train with current train_xgboost.py or set MODEL_FEATURES_PATH")
        report("saved XGBoost model loading", True, f"predict_proba available; {len(features)} ordered features")
    except Exception as exc:
        report("saved XGBoost model loading", False, f"{type(exc).__name__}: {exc}")


def check_api_endpoints(test_database_url: str) -> None:
    os.environ["DATABASE_URL"] = test_database_url
    # Ensure no backend modules retain an engine pointed at the live database.
    for name in ("main", "services", "database", "models"):
        sys.modules.pop(name, None)
    try:
        from fastapi.testclient import TestClient
        import main
        import services

        fake_quote = {
            "price": 2650.25, "bid": 2650.1, "ask": 2650.4, "spread": 0.3,
            "timestamp": datetime.now(timezone.utc), "source": "test", "is_stale": False,
        }
        with patch.object(main, "refresh_live_price", return_value=fake_quote), \
             patch.object(main, "create_prediction", return_value={"direction": "UP", "confidence": 0.72, "timestamp": datetime.now(timezone.utc).isoformat()}), \
             patch.object(main, "answer_query", return_value={"answer": "Test answer [1].", "sources": [{"title": "Test", "url": "https://example.org", "domain": "example.org", "published_at": None, "snippet": "test"}]}):
            with TestClient(main.app) as client:
                health = client.get("/health")
                report("GET /health", health.status_code == 200 and health.json().get("status") == "ok", f"HTTP {health.status_code}")

                live = client.get("/price/live")
                live_data = live.json()
                live_ok = live.status_code == 200 and {"price", "spread", "active_session", "is_stale"} <= live_data.keys()
                report("GET /price/live response", live_ok, f"HTTP {live.status_code}; mocked provider")

                history = client.get("/history")
                history_data = history.json()
                report("GET /history empty database", history.status_code == 200 and history_data.get("candles") == [], f"HTTP {history.status_code}")

                accuracy = client.get("/accuracy-log")
                accuracy_data = accuracy.json()
                accuracy_ok = (
                    accuracy.status_code == 200
                    and accuracy_data.get("accuracy_percent") is None
                    and accuracy_data.get("recent") == []
                    and accuracy_data.get("status") == "no_evaluated_predictions"
                )
                report("GET /accuracy-log empty-record grace", accuracy_ok, f"HTTP {accuracy.status_code}; no prediction rows")

                prediction = client.post("/predict")
                prediction_data = prediction.json()
                prediction_ok = prediction.status_code == 200 and prediction_data.get("direction") in {"UP", "DOWN", "NEUTRAL"} and 0 <= prediction_data.get("confidence", -1) <= 1
                report("POST /predict response", prediction_ok, f"HTTP {prediction.status_code}; inference mocked")

                chat_response = client.post("/chat", json={"query": "What is driving gold?", "history": []})
                chat_data = chat_response.json()
                chat_ok = chat_response.status_code == 200 and bool(chat_data.get("answer")) and isinstance(chat_data.get("sources"), list)
                report("POST /chat response", chat_ok, f"HTTP {chat_response.status_code}; RAG provider mocked")

                origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
                cors = client.options(
                    "/chat",
                    headers={"Origin": origins[0], "Access-Control-Request-Method": "POST"},
                )
                cors_ok = cors.status_code == 200 and cors.headers.get("access-control-allow-origin") == origins[0]
                report("CORS preflight", cors_ok, f"HTTP {cors.status_code}; origin={origins[0]}")

        # Prove recent database data is served stale and explicitly flagged if
        # both upstream price services fail/rate-limit.
        from database import SessionLocal
        from models import PriceHistory
        from sqlalchemy import select

        with SessionLocal() as session, session.begin():
            session.add(
                PriceHistory(
                    symbol="XAU/USD", timestamp=datetime.now(timezone.utc) - timedelta(seconds=20),
                    open=2649.0, high=2651.0, low=2648.0, close=2650.0, volume=None,
                    macro_features={}, feature_vector={},
                )
            )
        services._LAST_GOOD_QUOTE = None
        old_gold, old_twelve = os.getenv("GOLDAPI_API_KEY"), os.getenv("TWELVE_DATA_API_KEY")
        os.environ["GOLDAPI_API_KEY"] = "test-key"
        os.environ["TWELVE_DATA_API_KEY"] = "test-key"
        try:
            with patch.object(services, "_request_json", side_effect=services.ServiceUnavailable("simulated 429")):
                fallback = services.fetch_live_quote()
            report("price-provider outage fallback", fallback.get("is_stale") is True and fallback.get("source") == "database_stale" and fallback.get("price") == 2650.0, "recent database quote marked stale")
        finally:
            if old_gold is None:
                os.environ.pop("GOLDAPI_API_KEY", None)
            else:
                os.environ["GOLDAPI_API_KEY"] = old_gold
            if old_twelve is None:
                os.environ.pop("TWELVE_DATA_API_KEY", None)
            else:
                os.environ["TWELVE_DATA_API_KEY"] = old_twelve
    except Exception as exc:
        report("FastAPI integration checks", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true", help="Do not require/load a trained model artifact")
    parser.add_argument("--skip-keys", action="store_true", help="Skip API key-presence checks")
    args = parser.parse_args()

    original_database_url = os.getenv("DATABASE_URL")
    if not args.skip_keys:
        check_key_presence()
    check_live_database(original_database_url)
    check_feature_pipeline()

    with tempfile.TemporaryDirectory(prefix="gold-api-check-") as directory:
        test_db = Path(directory) / "api-test.sqlite3"
        test_database_url = f"sqlite+pysqlite:///{test_db.as_posix()}"
        os.environ["DATABASE_URL"] = test_database_url
        if not args.skip_model:
            check_model_artifact()
        check_api_endpoints(test_database_url)
    if original_database_url is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = original_database_url

    failed = [item for item in RESULTS if not item[1]]
    print(f"\nChecks: {len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
