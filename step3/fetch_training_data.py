"""Download XAU/USD intraday bars plus DXY and 10-year TIPS CSV inputs.

Reads TWELVE_DATA_API_KEY and optional TWELVE_DATA_DXY_SYMBOL from the
environment. FRED's public graph CSV is used for DFII10, so no FRED API key is
required. Twelve Data limits a response to 5,000 bars per symbol/request.

Run from this directory after loading the backend .env values:
  python fetch_training_data.py --output-dir data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TIMEOUT_SECONDS = 30
MAX_OUTPUTSIZE = 5000


class DataDownloadError(RuntimeError):
    """Raised when a market-data provider returns invalid or unavailable data."""


def _get_json(url: str, params: dict[str, str | int]) -> dict:
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        host = urlparse(url).hostname or "data provider"
        status = getattr(getattr(exc, "response", None), "status_code", None)
        status_text = f"HTTP {status}" if status is not None else type(exc).__name__
        raise DataDownloadError(f"Request to {host} failed ({status_text}).") from exc
    if not isinstance(payload, dict):
        raise DataDownloadError("Twelve Data returned an invalid response.")
    if payload.get("status") == "error" or payload.get("code"):
        message = str(payload.get("message", "provider rejected the request"))
        raise DataDownloadError(f"Twelve Data error: {message}")
    return payload


def _fetch_twelve_series(symbol: str, interval: str, api_key: str, outputsize: int) -> pd.DataFrame:
    payload = _get_json(
        TWELVE_DATA_URL,
        {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "order": "ASC",
            "timezone": "UTC",
            "apikey": api_key,
        },
    )
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise DataDownloadError(f"Twelve Data returned no bars for {symbol} at {interval}.")
    frame = pd.DataFrame(values).rename(columns={"datetime": "timestamp"})
    if "timestamp" not in frame or "close" not in frame:
        raise DataDownloadError(f"Twelve Data bars for {symbol} are missing timestamp or close.")
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if frame.empty:
        raise DataDownloadError(f"Twelve Data returned no valid OHLC bars for {symbol}.")
    return frame


def _fetch_tips() -> pd.DataFrame:
    try:
        response = requests.get(FRED_CSV_URL, params={"id": "DFII10"}, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        from io import StringIO

        frame = pd.read_csv(StringIO(response.text))
    except (requests.RequestException, pd.errors.ParserError, ValueError) as exc:
        raise DataDownloadError(f"FRED DFII10 download failed ({type(exc).__name__}).") from exc
    date_column = next((name for name in ("observation_date", "DATE") if name in frame), None)
    if date_column is None or "DFII10" not in frame:
        raise DataDownloadError("FRED DFII10 CSV did not contain its expected date/value columns.")
    frame = frame.rename(columns={date_column: "timestamp", "DFII10": "tips_yield"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["tips_yield"] = pd.to_numeric(frame["tips_yield"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "tips_yield"])
    return frame[["timestamp", "tips_yield"]].sort_values("timestamp").drop_duplicates("timestamp")


def fetch_training_data(output_dir: Path, outputsize: int) -> dict[str, int]:
    api_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not api_key:
        raise DataDownloadError("TWELVE_DATA_API_KEY is not set in the environment.")
    dxy_symbol = os.getenv("TWELVE_DATA_DXY_SYMBOL", "DXY").strip() or "DXY"

    try:
        gold = _fetch_twelve_series("XAU/USD", "5min", api_key, outputsize)
    except DataDownloadError as exc:
        raise DataDownloadError(f"XAU/USD 5-minute download failed: {exc}") from exc
    dxy = None
    try:
        dxy = _fetch_twelve_series(dxy_symbol, "1day", api_key, outputsize)
    except DataDownloadError as exc:
        print(
            f"Warning: DXY series {dxy_symbol!r} is unavailable from this Twelve Data account; "
            f"features will be built without DXY. Details: {exc}",
            file=sys.stderr,
        )
    tips = _fetch_tips()

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "xauusd_5m.csv": gold,
        "tips.csv": tips,
    }
    if dxy is not None:
        outputs["dxy.csv"] = dxy[["timestamp", "close"]].rename(columns={"close": "dxy_close"})
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return {filename: len(frame) for filename, frame in outputs.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--outputsize", type=int, default=MAX_OUTPUTSIZE)
    args = parser.parse_args()
    if not 1 <= args.outputsize <= MAX_OUTPUTSIZE:
        parser.error(f"--outputsize must be between 1 and {MAX_OUTPUTSIZE}")
    try:
        counts = fetch_training_data(args.output_dir, args.outputsize)
    except DataDownloadError as exc:
        print(f"Data download failed: {exc}", file=sys.stderr)
        return 1
    for filename, rows in counts.items():
        print(f"Wrote {rows:,} rows to {args.output_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
