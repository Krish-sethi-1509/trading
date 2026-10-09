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

def build_live_feature_snapshot(bars: pd.DataFrame, *, now: datetime | None = None, max_age_seconds: int | None = None):
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
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
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
    engineered = build_features(frame.reset_index(drop=True))
    latest = engineered.iloc[-1]
    excluded = {"timestamp", "target_timestamp", "future_timestamp", "target_class", "future_close", "future_return"}
    features = {}
    for name, value in latest.items():
        if name in excluded or not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        number = float(value)
        features[str(name)] = number if np.isfinite(number) else None
    return features, float(latest["close"]), last_timestamp

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
    return frame[["timestamp","open","high","low","close","volume"]].dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").drop_duplicates("timestamp",keep="last").reset_index(drop=True)

def fetch_live_feature_snapshot(*, now: datetime | None = None):
    """Fetch bars once every four minutes, then recompute the latest feature row."""
    global _BAR_CACHE
    ttl = max(0, int(os.getenv("LIVE_FEATURE_CACHE_SECONDS", "240")))
    with _BAR_CACHE_LOCK:
        if _BAR_CACHE is None or time.monotonic() - _BAR_CACHE[0] > ttl:
            _BAR_CACHE = (time.monotonic(), _download_recent_bars())
        bars = _BAR_CACHE[1].copy()
    return build_live_feature_snapshot(bars, now=now)
