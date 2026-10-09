"""Market data, feature loading, inference, and prediction scoring services."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from database import SessionLocal
from models import PriceHistory, PredictionLog
from live_features import LiveFeatureUnavailable, fetch_live_feature_snapshot

logger = logging.getLogger(__name__)
UTC = timezone.utc
GOLDAPI_URL = "https://www.goldapi.io/api/price/XAU/USD"
TWELVE_PRICE_URL = "https://api.twelvedata.com/price"
HTTP_TIMEOUT = float(os.getenv("MARKET_DATA_TIMEOUT_SECONDS", "12"))
_LAST_GOOD_QUOTE: dict[str, Any] | None = None


def _remember_quote(quote: dict[str, Any]) -> dict[str, Any]:
    global _LAST_GOOD_QUOTE
    quote["is_stale"] = False
    _LAST_GOOD_QUOTE = quote.copy()
    return quote


class ServiceUnavailable(RuntimeError):
    pass


def _resolve_backend_path(value: str | Path) -> Path:
    """Resolve configured relative paths from this backend directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _request_json(url: str, *, params: dict | None = None, headers: dict | None = None) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Never include URLs or headers in errors: Twelve Data keys are query
        # parameters and other provider tokens are authorization headers.
        host = urlparse(url).hostname or "market-data provider"
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning("Market-data request to %s failed (HTTP %s)", host, status or "network error")
        raise ServiceUnavailable(f"Market data provider {host} request failed.") from exc
    if not isinstance(payload, dict):
        raise ServiceUnavailable("Market data provider returned an invalid response")
    if payload.get("status") == "error" or payload.get("code"):
        host = urlparse(url).hostname or "market-data provider"
        raise ServiceUnavailable(f"Market data provider {host} returned an error response.")
    return payload


def fetch_live_quote() -> dict[str, Any]:
    """Try configured providers in order, then a recent last-known database quote."""
    cache_ttl = max(0, int(os.getenv("PRICE_CACHE_SECONDS", "15")))
    if _LAST_GOOD_QUOTE is not None and cache_ttl:
        age = (utc_now() - _LAST_GOOD_QUOTE["timestamp"]).total_seconds()
        if 0 <= age <= cache_ttl:
            return {**_LAST_GOOD_QUOTE, "source": f"cached_{_LAST_GOOD_QUOTE['source']}"}
    failures: list[str] = []
    goldapi_key = os.getenv("GOLDAPI_API_KEY") or os.getenv("GOLD_API_KEY")
    if goldapi_key:
        try:
            payload = _request_json(GOLDAPI_URL, headers={"x-access-token": goldapi_key})
            price = float(payload["price"])
            bid = float(payload["bid"])
            ask = float(payload["ask"])
            quoted_at = datetime.fromtimestamp(float(payload["timestamp"]), UTC)
            if price <= 0 or bid <= 0 or ask < bid:
                raise ValueError("invalid quote values")
            return _remember_quote({
                "price": price, "bid": bid, "ask": ask, "spread": ask - bid,
                "timestamp": quoted_at, "source": "goldapi",
            })
        except (ServiceUnavailable, KeyError, TypeError, ValueError) as exc:
            failures.append(f"GoldAPI: {exc}")
            logger.warning("GoldAPI quote failed; trying the configured fallback provider")

    twelve_key = os.getenv("TWELVE_DATA_API_KEY")
    if twelve_key:
        try:
            payload = _request_json(
                TWELVE_PRICE_URL,
                params={"symbol": "XAU/USD", "apikey": twelve_key},
            )
            price = float(payload["price"])
            if price <= 0:
                raise ValueError("price must be positive")
            return _remember_quote({
                "price": price, "bid": None, "ask": None, "spread": None,
                "timestamp": utc_now(), "source": "twelve_data",
            })
        except (ServiceUnavailable, KeyError, TypeError, ValueError) as exc:
            failures.append(f"Twelve Data: {exc}")
            logger.warning("Twelve Data quote failed")

    # The process cache protects against short provider outages and avoids
    # repeated API calls across simultaneous dashboard clients.
    max_stale_seconds = int(os.getenv("PRICE_FALLBACK_MAX_STALE_SECONDS", "900"))
    if _LAST_GOOD_QUOTE is not None:
        age = (utc_now() - _LAST_GOOD_QUOTE["timestamp"]).total_seconds()
        if 0 <= age <= max_stale_seconds:
            return {**_LAST_GOOD_QUOTE, "is_stale": True, "source": f"cached_{_LAST_GOOD_QUOTE['source']}"}

    # Cross-worker/restart grace: use only a recent persisted quote and clearly
    # label it stale. A stale price is never represented as a live observation.
    try:
        with SessionLocal() as session:
            latest = session.scalar(
                select(PriceHistory)
                .where(PriceHistory.symbol == "XAU/USD")
                .order_by(PriceHistory.timestamp.desc())
                .limit(1)
            )
            if latest is not None:
                latest_time = latest.timestamp
                if latest_time.tzinfo is None:
                    latest_time = latest_time.replace(tzinfo=UTC)
                age = (utc_now() - latest_time.astimezone(UTC)).total_seconds()
                if 0 <= age <= max_stale_seconds:
                    return {
                        "price": latest.close, "bid": None, "ask": None, "spread": None,
                        "timestamp": latest_time.astimezone(UTC), "source": "database_stale",
                        "is_stale": True,
                    }
    except Exception as exc:
        logger.warning("Could not read last-known price from database: %s", exc)

    if failures:
        logger.error("All configured XAU/USD price providers failed: %s", "; ".join(failures))
        raise ServiceUnavailable("Price providers are unavailable and no recent cached quote is available.")
    raise ServiceUnavailable("Configure GOLD_API_KEY (or GOLDAPI_API_KEY) or TWELVE_DATA_API_KEY to enable live prices.")


def active_session(timestamp: datetime | None = None) -> str:
    """Approximate major FX sessions using UTC wall-clock hours."""
    current = timestamp or utc_now()
    hour = current.astimezone(UTC).hour
    if 12 <= hour < 16:
        return "LONDON_NEW_YORK_OVERLAP"
    if 7 <= hour < 16:
        return "LONDON"
    if 12 <= hour < 21:
        return "NEW_YORK"
    if 0 <= hour < 9:
        return "ASIA"
    return "OFF_HOURS"


def refresh_live_price() -> dict[str, Any]:
    """Fetch and idempotently store the latest quote as a minute price point."""
    quote = fetch_live_quote()
    timestamp = quote["timestamp"].replace(second=0, microsecond=0)
    # Spot quote APIs do not necessarily publish volume or candle OHLC. Do not
    # fabricate these: a quote is represented as a flat OHLC minute bar.
    # Minute spot quotes are not OHLCV candles. Build model inputs from recent
    # completed 5-minute provider bars and retain the true input timestamp.
    feature_vector: dict[str, Any] = {}
    try:
        features, feature_price, feature_timestamp = fetch_live_feature_snapshot()
        feature_vector = {
            **features,
            "_source_timestamp": feature_timestamp.isoformat(),
            "_source_price": feature_price,
            "_source": "twelve_data_5min",
        }
    except LiveFeatureUnavailable as exc:
        logger.warning("Live engineered features unavailable: %s", exc)

    row = {
        "symbol": "XAU/USD",
        "timestamp": timestamp,
        "open": quote["price"],
        "high": quote["price"],
        "low": quote["price"],
        "close": quote["price"],
        "volume": None,
        "macro_features": {},
        "feature_vector": feature_vector,
    }
    # Keep the quote endpoint usable during a transient database outage too;
    # callers receive the observed quote, while persistence failure is logged.
    try:
        with SessionLocal() as session, session.begin():
            statement = pg_insert(PriceHistory).values(**row)
            statement = statement.on_conflict_do_update(
                index_elements=[PriceHistory.symbol, PriceHistory.timestamp],
                set_={
                    "close": statement.excluded.close,
                    "high": func.greatest(PriceHistory.high, statement.excluded.high),
                    "low": func.least(PriceHistory.low, statement.excluded.low),
                    "feature_vector": statement.excluded.feature_vector,
                },
            )
            session.execute(statement)
    except Exception:
        logger.exception("Live quote was fetched but could not be persisted")
        quote = {**quote, "persistence_warning": True}
    return quote


_MODEL_CACHE: dict[str, Any] = {}


def _load_artifacts() -> tuple[Any, Any | None, list[str] | None]:
    default_model = Path(__file__).resolve().parent.parent / "step3/artifacts/xgb/xgboost_pipeline.joblib"
    model_path = _resolve_backend_path(os.getenv("MODEL_PATH", str(default_model)))
    scaler_path_value = os.getenv("SCALER_PATH")
    scaler_path = _resolve_backend_path(scaler_path_value) if scaler_path_value else None
    feature_path_value = os.getenv("MODEL_FEATURES_PATH")
    feature_path = _resolve_backend_path(feature_path_value) if feature_path_value else None
    artifact_paths = [model_path] + ([scaler_path] if scaler_path else []) + ([feature_path] if feature_path else [])
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        raise ServiceUnavailable("Model artifact missing: " + ", ".join(missing))
    signature = tuple((str(path), path.stat().st_mtime_ns) for path in artifact_paths)
    if _MODEL_CACHE.get("signature") == signature:
        return _MODEL_CACHE["model"], _MODEL_CACHE.get("scaler"), _MODEL_CACHE.get("features")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    feature_names = None
    if feature_path:
        feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
        if not isinstance(feature_names, list) or not all(isinstance(item, str) for item in feature_names):
            raise ServiceUnavailable("MODEL_FEATURES_PATH must contain a JSON array of feature names")
    else:
        feature_names = getattr(model, "gold_feature_columns_", None)
    _MODEL_CACHE.update(signature=signature, model=model, scaler=scaler, features=feature_names)
    return model, scaler, feature_names


def _latest_feature_vector(session: Session) -> tuple[dict[str, Any], float, datetime]:
    """Return a persisted live feature snapshot; never fall back to historical CSV data."""
    rows = session.scalars(
        select(PriceHistory).where(PriceHistory.symbol == "XAU/USD")
        .order_by(PriceHistory.timestamp.desc()).limit(100)
    ).all()
    for row in rows:
        stored = row.feature_vector or {}
        if not stored.get("_source_timestamp") or stored.get("_source_price") is None:
            continue
        try:
            timestamp = pd.to_datetime(stored["_source_timestamp"], utc=True, errors="raise").to_pydatetime()
            features = {key: value for key, value in stored.items() if not key.startswith("_")}
            return features, float(stored["_source_price"]), timestamp
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed live feature snapshot at %s", row.timestamp)
    raise ServiceUnavailable(
        "No engineered live feature vector is stored. The scheduler must fetch recent completed "
        "5-minute XAU/USD candles using TWELVE_DATA_API_KEY before prediction."
    )

def _prepare_feature_frame(features: dict[str, Any], model: Any, feature_names: list[str] | None) -> pd.DataFrame:
    if feature_names is None:
        feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise ServiceUnavailable("Model has no stored feature order; set MODEL_FEATURES_PATH")
    try:
        # Missing or undefined rolling features stay NaN so the pipeline's
        # fitted imputer can handle warm-up windows and unavailable indicators.
        return pd.DataFrame(
            [[float(features[name]) if features.get(name) is not None else np.nan for name in feature_names]],
            columns=feature_names,
        )
    except (TypeError, ValueError) as exc:
        raise ServiceUnavailable(f"Latest feature vector contains nonnumeric inputs: {exc}") from exc


def create_prediction(session: Session) -> dict[str, Any]:
    """Run inference, persist a prediction, and return the API response shape."""
    features, reference_price, feature_timestamp = _latest_feature_vector(session)
    feature_timestamp = _as_utc(feature_timestamp)
    max_age = max(0, int(os.getenv("FEATURE_MAX_AGE_SECONDS", "900")))
    feature_age = (utc_now() - feature_timestamp).total_seconds()
    if feature_age < 0 or feature_age > max_age:
        raise ServiceUnavailable(
            f"Latest engineered features are stale ({max(0, int(feature_age))} seconds old; "
            f"maximum is {max_age}). Refresh the OHLCV feature pipeline before predicting."
        )
    model, scaler, feature_names = _load_artifacts()
    model_input = _prepare_feature_frame(features, model, feature_names)
    try:
        if scaler is not None:
            transformed = scaler.transform(model_input)
            probabilities = model.predict_proba(transformed)[0]
            model_classes = getattr(model, "classes_", np.arange(len(probabilities)))
            trained_ids = None
        else:
            probabilities = model.predict_proba(model_input)[0]
            model_classes = getattr(model, "classes_", np.arange(len(probabilities)))
            trained_ids = getattr(model, "gold_trained_class_ids_", None)
    except Exception as exc:
        raise ServiceUnavailable(f"Model inference failed: {exc}") from exc
    if trained_ids is not None:
        id_by_model_class = {index: actual_id for index, actual_id in enumerate(trained_ids)}
    else:
        id_by_model_class = {int(class_id): int(class_id) for class_id in model_classes}
    probabilities_by_direction: dict[str, float] = {}
    direction_by_id = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
    for position, model_class in enumerate(model_classes):
        class_id = id_by_model_class.get(int(model_class), int(model_class))
        direction = direction_by_id.get(class_id)
        if direction:
            probabilities_by_direction[direction] = float(probabilities[position])
    if not probabilities_by_direction:
        raise ServiceUnavailable("Model classes could not be mapped to DOWN/NEUTRAL/UP")
    direction = max(probabilities_by_direction, key=probabilities_by_direction.get)
    confidence = probabilities_by_direction[direction]
    predicted_at = utc_now()
    target_hours = float(os.getenv("PREDICTION_HORIZON_HOURS", "4"))
    entry = PredictionLog(
        timestamp=predicted_at,
        target_timestamp=predicted_at + timedelta(hours=target_hours),
        predicted_direction=direction,
        confidence_score=confidence,
        reference_price=reference_price,
    )
    session.add(entry)
    session.commit()
    return {"direction": direction, "confidence": confidence, "timestamp": predicted_at.isoformat()}


def update_prediction_outcomes(session: Session) -> None:
    """Score due predictions against the closest historical close after target time."""
    now = utc_now()
    pending = session.scalars(
        select(PredictionLog).where(
            PredictionLog.actualized_at.is_(None), PredictionLog.target_timestamp <= now
        )
    ).all()
    for prediction in pending:
        observed = session.scalar(
            select(PriceHistory)
            .where(
                PriceHistory.symbol == "XAU/USD",
                PriceHistory.timestamp >= prediction.target_timestamp,
            )
            .order_by(PriceHistory.timestamp.asc())
            .limit(1)
        )
        if observed is None:
            continue
        max_delay = max(0, int(os.getenv("OUTCOME_MAX_DELAY_SECONDS", "900")))
        delay = (_as_utc(observed.timestamp) - _as_utc(prediction.target_timestamp)).total_seconds()
        if delay < 0 or delay > max_delay:
            # Leave the record pending instead of scoring against a price hours
            # or days after its target (for example, across a weekend closure).
            continue
        change = observed.close / prediction.reference_price - 1.0
        threshold = float(os.getenv("NEUTRAL_RETURN_THRESHOLD", "0.001"))
        actual = "UP" if change > threshold else "DOWN" if change < -threshold else "NEUTRAL"
        prediction.actual_outcome = actual
        prediction.accuracy_flag = prediction.predicted_direction == actual
        prediction.actualized_at = now
    if pending:
        session.commit()


def run_scheduled_prediction() -> dict[str, Any]:
    """Scheduler entry point that shares the same inference/logging service."""
    with SessionLocal() as session:
        return create_prediction(session)


def run_scheduled_refresh() -> dict[str, Any]:
    """Scheduler entry point; failures are raised for scheduler logging."""
    return refresh_live_price()
