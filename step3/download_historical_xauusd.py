"""Download and resample Dukascopy XAU/USD bid/ask minute candles.

Dukascopy BI5 bars are a single broker feed. Their volume field is a feed
activity measure, not consolidated OTC gold trading volume. This downloader
forms midpoint OHLC and averages the two side volumes; document that limitation
when interpreting volume-dependent features.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import lzma
import struct
import time
import warnings
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">IIIIIf")
PRICE_SCALE = 1000


def decode_bi5(payload: bytes, day: date) -> pd.DataFrame:
    """Decode one side's 24-byte-per-candle LZMA-alone BI5 payload."""
    if not payload:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw = lzma.decompress(payload, format=lzma.FORMAT_ALONE)
    if len(raw) % RECORD.size:
        raise ValueError(f"Malformed BI5 payload: {len(raw)} bytes is not divisible by 24")
    rows = []
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    for offset, open_, close, low, high, volume in RECORD.iter_unpack(raw):
        rows.append((
            midnight + timedelta(seconds=offset),
            open_ / PRICE_SCALE, high / PRICE_SCALE, low / PRICE_SCALE,
            close / PRICE_SCALE, float(volume),
        ))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_side(day: date, side: str, *, retries: int = 1) -> pd.DataFrame:
    """Fetch one UTC day's BID or ASK candles; missing market days return empty."""
    month_index = day.month - 1
    url = (f"{BASE_URL}/XAUUSD/{day.year}/{month_index:02d}/{day.day:02d}/"
           f"{side}_candles_min_1.bi5")
    request = Request(url, headers={"User-Agent": "xauusd-research/1.0"})
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=8) as response:
                return decode_bi5(response.read(), day)
        except HTTPError as exc:
            if exc.code == 404:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt + 1 == retries:
                warnings.warn(f"Skipping {side} {day}: HTTP {exc.code}", RuntimeWarning)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        except (TimeoutError, URLError) as exc:
            if attempt + 1 == retries:
                warnings.warn(f"Skipping {side} {day} after network timeout: {exc}", RuntimeWarning)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        time.sleep(0.5 * (2 ** attempt))
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def midpoint_bars(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Match both quote sides by timestamp and form midpoint OHLC bars."""
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if bid.empty or ask.empty:
        return pd.DataFrame(columns=columns)
    b = bid.set_index("timestamp").sort_index()
    a = ask.set_index("timestamp").sort_index()
    common = b.index.intersection(a.index)
    if common.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(index=common)
    for field in ("open", "high", "low", "close"):
        result[field] = (b.loc[common, field].astype(float) + a.loc[common, field].astype(float)) / 2
    result["volume"] = (b.loc[common, "volume"].astype(float) + a.loc[common, "volume"].astype(float)) / 2
    result.index.name = "timestamp"
    return result.reset_index()[columns]


def resample_five_minutes(minutes: pd.DataFrame, minimum_minutes: int = 3) -> pd.DataFrame:
    """Resample complete midpoint minutes; drop sparse 5-minute intervals."""
    if minutes.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = minutes.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    aggregation = frame.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), minute_count=("close", "count"),
    )
    aggregation = aggregation.loc[aggregation["minute_count"] >= minimum_minutes]
    return aggregation.drop(columns="minute_count").reset_index()


def download_history(start: date, end: date, workers: int = 6) -> pd.DataFrame:
    """Download both quote sides for inclusive UTC date range and return 5-minute bars."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    downloaded: dict[tuple[date, str], pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_side, day, side): (day, side) for day in days for side in ("BID", "ASK")}
        for future in as_completed(futures):
            downloaded[futures[future]] = future.result()
    daily = []
    for day in days:
        daily.append(midpoint_bars(downloaded[(day, "BID")], downloaded[(day, "ASK")]))
    minute_bars = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    result = resample_five_minutes(minute_bars)
    if len(result) < 1000:
        raise RuntimeError(f"Only {len(result)} matched 5-minute bars were downloaded; at least 1,000 are required to evaluate")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    bars = download_history(args.start_date, args.end_date, args.workers)
    bars.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Saved {len(bars):,} five-minute midpoint bars from {bars.timestamp.min()} to {bars.timestamp.max()} to {args.output}")
    print("Source: Dukascopy; volume is a broker-feed activity proxy, not consolidated OTC volume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
