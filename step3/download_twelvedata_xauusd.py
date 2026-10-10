"""Download a coverage-checked six-month XAU/USD 5-minute series from Twelve Data.

The API caps each response at 5,000 bars. This downloader requests small,
non-overlapping UTC date windows, retries transient provider/network failures,
deduplicates boundary timestamps, and fails if date or bar coverage is short.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time as dt_time, timedelta, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.twelvedata.com/time_series"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DataDownloadError(RuntimeError):
    """Raised when the provider returns unavailable or invalid market data."""


def fetch_window(
    start: datetime,
    end: datetime,
    api_key: str,
    *,
    retries: int = 6,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one short UTC range, retrying temporary failures and rejecting errors."""
    client = session or requests
    params = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "order": "ASC",
        "timezone": "UTC",
        "apikey": api_key,
    }
    for attempt in range(retries):
        try:
            response = client.get(API_URL, params=params, timeout=30)
            if response.status_code in RETRYABLE_STATUS:
                if attempt + 1 == retries:
                    raise DataDownloadError(
                        f"Twelve Data remained unavailable (HTTP {response.status_code}) "
                        f"for {params['start_date']} through {params['end_date']}."
                    )
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(60.0, max(float(retry_after), 1.0)) if retry_after else min(30.0, 2.0 ** attempt)
                except ValueError:
                    delay = min(30.0, 2.0 ** attempt)
                time.sleep(delay + random.uniform(0.0, 0.5))
                continue
            response.raise_for_status()
            payload = response.json()
        except DataDownloadError:
            raise
        except (requests.RequestException, ValueError) as exc:
            if attempt + 1 == retries:
                raise DataDownloadError(
                    f"Twelve Data request failed for {params['start_date']} through "
                    f"{params['end_date']} ({type(exc).__name__})."
                ) from exc
            time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0.0, 0.5))
            continue

        if not isinstance(payload, dict):
            raise DataDownloadError("Twelve Data returned a malformed response.")
        if payload.get("status") == "error" or payload.get("code"):
            code = payload.get("code")
            if code in RETRYABLE_STATUS and attempt + 1 < retries:
                time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0.0, 0.5))
                continue
            raise DataDownloadError(
                f"Twelve Data error for {params['start_date']} through {params['end_date']}: "
                f"{payload.get('message', 'unknown provider error')}."
            )
        values = payload.get("values")
        if not isinstance(values, list):
            raise DataDownloadError("Twelve Data response has no values list.")
        if not values:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        frame = pd.DataFrame(values).rename(columns={"datetime": "timestamp"})
        if "timestamp" not in frame:
            raise DataDownloadError("Twelve Data bars are missing timestamps.")
        for column in OUTPUT_COLUMNS[1:]:
            if column not in frame:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        frame = frame[OUTPUT_COLUMNS].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        return frame.reset_index(drop=True)
    raise DataDownloadError("Twelve Data retries were exhausted.")


def coverage_report(
    bars: pd.DataFrame,
    start: date,
    end: date,
    *,
    min_bars_per_day: int = 200,
    min_coverage: float = 0.90,
) -> dict[str, object]:
    """Measure requested range, observed endpoints, weekdays, and expected bar count."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    weekdays = [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]
    if bars.empty:
        first = last = None
        counts = pd.Series(dtype="int64")
    else:
        ts = pd.to_datetime(bars["timestamp"], utc=True, errors="raise")
        first, last = ts.min(), ts.max()
        counts = ts.dt.date.value_counts()
    short_days = [d.isoformat() for d in weekdays if int(counts.get(d, 0)) < min_bars_per_day]
    expected = len(weekdays) * 276
    actual = int(len(bars))
    ratio = actual / expected if expected else 0.0
    first_expected = weekdays[0] if weekdays else start
    last_expected = weekdays[-1] if weekdays else end
    endpoints_ok = bool(
        first is not None
        and start <= first.date() <= first_expected
        and last_expected <= last.date() <= end
    )
    return {
        "provider": "Twelve Data",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "actual_first_timestamp": first.isoformat() if first is not None else None,
        "actual_last_timestamp": last.isoformat() if last is not None else None,
        "expected_weekdays": len(weekdays),
        "minimum_bars_per_day": min_bars_per_day,
        "missing_or_short_weekdays": short_days,
        "actual_5m_bars": actual,
        "expected_5m_bars_at_23h_per_weekday": int(expected),
        "bar_coverage": float(ratio),
        "endpoint_coverage_ok": endpoints_ok,
        "complete": bool(not short_days and ratio >= min_coverage and endpoints_ok),
    }


def download_history(start: date, end: date, api_key: str) -> pd.DataFrame:
    """Download 5-day UTC chunks, staying below Twelve Data's 5,000-bar response cap."""
    if not api_key:
        raise DataDownloadError("TWELVE_DATA_API_KEY is not set.")
    if end < start:
        raise ValueError("end date must be on or after start date")
    start_dt = datetime.combine(start, dt_time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end, dt_time.max.replace(microsecond=0), tzinfo=timezone.utc)
    frames = []
    cursor = start_dt
    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=5) - timedelta(seconds=1), end_dt)
        chunk = fetch_window(cursor, window_end, api_key)
        if not chunk.empty:
            frames.append(chunk)
        cursor = window_end + timedelta(seconds=1)
        # Respect provider request quotas while keeping the full range practical.
        time.sleep(0.25)
    bars = (
        pd.concat(frames, ignore_index=True)
        if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    bars = bars.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if not bars.empty:
        bars["volume"] = np.nan  # Live inference also uses volume-free features.
    return bars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    args = parser.parse_args()
    try:
        bars = download_history(args.start_date, args.end_date, os.getenv("TWELVE_DATA_API_KEY", "").strip())
        report = coverage_report(bars, args.start_date, args.end_date)
        args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        if not report["complete"]:
            raise DataDownloadError("Incomplete Twelve Data history; refusing to train on a short range.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(args.output, index=False, float_format="%.6f", date_format="%Y-%m-%dT%H:%M:%SZ")
        print(f"Saved {len(bars):,} five-minute XAU/USD bars to {args.output}.")
    except (DataDownloadError, requests.RequestException, ValueError) as exc:
        print(f"Historical download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
