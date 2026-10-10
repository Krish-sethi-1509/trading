"""Download and resample Dukascopy XAU/USD bid/ask minute candles.

Dukascopy BI5 bars are a single broker feed. Their volume field is a feed
activity measure, not consolidated OTC gold trading volume. This downloader
forms midpoint OHLC and averages the two side volumes; document that limitation
when interpreting volume-dependent features.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import lzma
import random
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


def fetch_side(day: date, side: str, *, retries: int = 7) -> pd.DataFrame:
    """Fetch one UTC day's BID or ASK candles; missing market days return empty."""
    month_index = day.month - 1
    url = (f"{BASE_URL}/XAUUSD/{day.year}/{month_index:02d}/{day.day:02d}/"
           f"{side}_candles_min_1.bi5")
    request = Request(url, headers={"User-Agent": "xauusd-research/1.0"})
    for attempt in range(retries):
        retry_delay = min(30.0, 1.0 * (2 ** attempt))
        try:
            with urlopen(request, timeout=25) as response:
                return decode_bi5(response.read(), day)
        except HTTPError as exc:
            if exc.code == 404:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after:
                try:
                    retry_delay = min(60.0, max(retry_delay, float(retry_after)))
                except ValueError:
                    pass
            if attempt + 1 == retries:
                warnings.warn(f"Skipping {side} {day}: HTTP {exc.code}", RuntimeWarning)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        except (TimeoutError, URLError, OSError) as exc:
            if attempt + 1 == retries:
                warnings.warn(f"Skipping {side} {day} after network timeout: {exc}", RuntimeWarning)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        time.sleep(retry_delay + random.uniform(0.0, 0.5))
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


def validate_history_coverage(
    bars: pd.DataFrame,
    start: date,
    end: date,
    *,
    min_bars_per_day: int = 200,
    min_total_bar_coverage: float = 0.90,
) -> dict[str, object]:
    """Fail closed unless requested weekdays and expected intraday bars are present."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    if bars.empty or "timestamp" not in bars:
        raise RuntimeError("Historical download returned no timestamped 5-minute bars")
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="raise")
    first_timestamp, last_timestamp = timestamps.min(), timestamps.max()
    expected_days = [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
        if (start + timedelta(days=offset)).weekday() < 5
    ]
    day_counts = timestamps.dt.date.value_counts()
    missing_days = [
        day.isoformat() for day in expected_days
        if int(day_counts.get(day, 0)) < min_bars_per_day
    ]
    # Dukascopy XAU/USD normally publishes about 23 hours per weekday after
    # the daily maintenance break: 276 five-minute bars.
    expected_bars = len(expected_days) * 276
    bar_coverage = len(bars) / expected_bars if expected_bars else 0.0
    first_expected = expected_days[0] if expected_days else start
    last_expected = expected_days[-1] if expected_days else end
    first_day, last_day = first_timestamp.date(), last_timestamp.date()
    endpoints_ok = (
        start <= first_day <= first_expected
        and last_expected <= last_day <= end
    )
    report = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "actual_first_timestamp": first_timestamp.isoformat(),
        "actual_last_timestamp": last_timestamp.isoformat(),
        "expected_weekdays": len(expected_days),
        "minimum_bars_per_day": min_bars_per_day,
        "missing_or_short_weekdays": missing_days,
        "actual_5m_bars": int(len(bars)),
        "expected_5m_bars_at_23h_per_weekday": int(expected_bars),
        "bar_coverage": float(bar_coverage),
        "endpoint_coverage_ok": bool(endpoints_ok),
    }
    failures = []
    if missing_days:
        failures.append(f"{len(missing_days)} weekdays are missing or have fewer than {min_bars_per_day} bars")
    if bar_coverage < min_total_bar_coverage:
        failures.append(f"bar coverage {bar_coverage:.1%} is below {min_total_bar_coverage:.1%}")
    if not endpoints_ok:
        failures.append(
            f"actual dates {first_day} through {last_day} do not cover expected endpoints "
            f"{first_expected} through {last_expected}"
        )
    if failures:
        raise RuntimeError(
            "Incomplete historical XAU/USD dataset: " + "; ".join(failures)
            + ". Coverage report: " + json.dumps(report, sort_keys=True)
        )
    return report


def download_history(start: date, end: date, workers: int = 6) -> pd.DataFrame:
    """Download both quote sides for inclusive UTC date range and return validated 5-minute bars."""
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
    coverage = validate_history_coverage(result, start, end)
    print("Historical coverage: " + json.dumps(coverage, sort_keys=True))
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage-report", help="Optional path for requested-vs-actual coverage JSON")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--omit-volume", action="store_true", help="Write volume as missing to match live feeds without volume")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    bars = download_history(args.start_date, args.end_date, args.workers)
    coverage = validate_history_coverage(bars, args.start_date, args.end_date)
    if args.omit_volume:
        bars["volume"] = np.nan
    bars.to_csv(args.output, index=False, float_format="%.6f")
    if args.coverage_report:
        from pathlib import Path
        report_path = Path(args.coverage_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"Saved {len(bars):,} five-minute midpoint bars from {bars.timestamp.min()} to {bars.timestamp.max()} to {args.output}")
    print("Source: Dukascopy midpoint bars; volume is a broker-feed activity proxy, not consolidated OTC volume.")
    if args.omit_volume:
        print("Volume omitted to match live inference feeds without volume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
