"""Fetch recent provider OHLCV candles and build causal training-compatible features."""
from __future__ import annotations
import os, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from step3.feature_engineering import build_features  # noqa: E402

UTC = timezone.utc
TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
_BAR_CACHE = None
_BAR_CACHE_LOCK = threading.Lock()

class LiveFeatureUnavailable(RuntimeError):
    """Fresh, compatible live bars could not be obtained or engineered."""

def build_live_feature_snapshot(bars: pd.DataFrame, *, tips: pd.DataFrame | None = None, now: datetime | None = None, max_age_seconds: int | None = None):
    """Engineer a vector from completed 5-minute bars using the training feature builder."""
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    current = current.astimezone(UTC)
    max_age = max_age_seconds if max_age_seconds is not None else int(os.getenv("FEATURE_MAX_AGE_SECONDS", "900"))
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise LiveFeatureUnavailable(f"Live candle response is missing: {', '.join(sorted(missing))}")
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame.empty or frame[["timestamp", "open", "high", "low", "close"]].isna().any().any():
        raise LiveFeatureUnavailable("Twelve Data returned incomplete timestamps or OHLC candles.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise LiveFeatureUnavailable("Twelve Data returned non-positive OHLC values.")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any() or (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise LiveFeatureUnavailable("Twelve Data returned inconsistent OHLC values.")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame = frame.loc[frame["timestamp"] + pd.Timedelta(minutes=5) <= pd.Timestamp(current)]
    if frame.empty:
        raise LiveFeatureUnavailable("Provider returned no completed 5-minute XAU/USD candles.")
    last_timestamp = frame.iloc[-1]["timestamp"].to_pydatetime()
    age = (current - last_timestamp).total_seconds()
    if age < 0 or age > max_age:
        raise LiveFeatureUnavailable(f"Latest completed 5-minute candle is {max(0, int(age))} seconds old; maximum allowed is {max_age}.")
    if len(frame) < 50:
        raise LiveFeatureUnavailable(f"Need at least 50 completed 5-minute candles to warm up rolling features; got {len(frame)}.")
    recent_deltas = frame["timestamp"].tail(12).diff().dropna().dt.total_seconds()
    if len(recent_deltas) < 11 or (recent_deltas > 7 * 60).any() or (recent_deltas < 5 * 60).any():
        raise LiveFeatureUnavailable("Recent provider candles are incomplete or not on a continuous 5-minute grid.")
    try:
        engineered = build_features(frame.reset_index(drop=True), tips=tips, use_volume=False)
    except (ValueError, KeyError, TypeError) as exc:
        raise LiveFeatureUnavailable(f"Could not engineer features from the latest live candles: {exc}") from exc
    latest = engineered.iloc[-1]
    excluded = {"timestamp", "target_timestamp", "future_timestamp", "target_class", "future_close", "future_return"}
    features = {}
    for name, value in latest.items():
        if name in excluded or not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        number = float(value)
        features[str(name)] = number if np.isfinite(number) else None
    return features, float(latest["close"]), last_timestamp

def _download_recent_tips(*, now: datetime | None = None) -> pd.DataFrame:
    """Fetch published DFII10 observations; feature_engineering applies its one-day release lag."""
    from io import StringIO
    current = now or datetime.now(UTC)
    try:
        response = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": "DFII10"},
            timeout=float(os.getenv("MARKET_DATA_TIMEOUT_SECONDS", "12")),
        )
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
    except (requests.RequestException, pd.errors.ParserError, ValueError) as exc:
        raise LiveFeatureUnavailable("Could not fetch current FRED DFII10 observations for live features.") from exc
    date_column = next((name for name in ("observation_date", "DATE") if name in raw), None)
    value_column = "DFII10" if "DFII10" in raw else None
    if date_column is None or value_column is None:
        raise LiveFeatureUnavailable("FRED returned an unexpected DFII10 CSV schema.")
    tips = raw.rename(columns={date_column: "timestamp", value_column: "tips_yield"})[["timestamp", "tips_yield"]]
    tips["timestamp"] = pd.to_datetime(tips["timestamp"], utc=True, errors="coerce")
    tips["tips_yield"] = pd.to_numeric(tips["tips_yield"], errors="coerce")
    tips = tips.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if tips.empty or (current.astimezone(UTC) - tips.iloc[-1]["timestamp"].to_pydatetime()).total_seconds() > 5 * 86400:
        raise LiveFeatureUnavailable("FRED DFII10 observations are missing or older than five days.")
    return tips.reset_index(drop=True)


def _download_recent_bars() -> pd.DataFrame:
    key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not key:
        raise LiveFeatureUnavailable("TWELVE_DATA_API_KEY is required for live 5-minute OHLCV prediction features.")
    outputsize = min(5000, max(100, int(os.getenv("LIVE_FEATURE_BAR_COUNT", "500"))))
    try:
        response = requests.get(TWELVE_DATA_URL, params={"symbol":"XAU/USD","interval":"5min","outputsize":outputsize,"order":"ASC","timezone":"UTC","apikey":key}, timeout=float(os.getenv("MARKET_DATA_TIMEOUT_SECONDS","12")))
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise LiveFeatureUnavailable(f"Twelve Data live-candle request failed (HTTP {status or 'network error'}).") from exc
    if not isinstance(payload, dict) or payload.get("status") == "error" or payload.get("code"):
        raise LiveFeatureUnavailable("Twelve Data could not provide live XAU/USD candles.")
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise LiveFeatureUnavailable("Twelve Data returned no live XAU/USD candles.")
    frame = pd.DataFrame(values).rename(columns={"datetime":"timestamp"})
    for name in ("open","high","low","close","volume"):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame.empty or frame[["timestamp","open","high","low","close"]].isna().any().any():
        raise LiveFeatureUnavailable("Twelve Data returned incomplete timestamps or OHLC candles.")
    return frame[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").drop_duplicates("timestamp",keep="last").reset_index(drop=True)

def fetch_live_feature_snapshot(*, now: datetime | None = None):
    """Fetch bars once every four minutes, then recompute the latest feature row."""
    global _BAR_CACHE
    ttl = max(0, int(os.getenv("LIVE_FEATURE_CACHE_SECONDS", "240")))
    with _BAR_CACHE_LOCK:
        if _BAR_CACHE is None or time.monotonic() - _BAR_CACHE[0] > ttl:
            _BAR_CACHE = (time.monotonic(), _download_recent_bars(), _download_recent_tips(now=now))
        bars, tips = _BAR_CACHE[1].copy(), _BAR_CACHE[2].copy()
    return build_live_feature_snapshot(bars, tips=tips, now=now)
