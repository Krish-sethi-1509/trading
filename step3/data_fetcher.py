"""Fetch daily gold, DXY, and 10-year TIPS yield data and persist it in PostgreSQL.

Configuration is supplied with environment variables:
  DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/gold_mvp
  TWELVE_DATA_API_KEY=...
  FRED_API_KEY=...

Twelve Data supplies XAU/USD and DXY daily bars. FRED series DFII10 is the
market yield on 10-year Treasury inflation-indexed securities, in percent.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import Any

import requests
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import EconomicObservation, MarketBar

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
HTTP_TIMEOUT_SECONDS = 30


class DataProviderError(RuntimeError):
    """Raised when a provider returns an invalid or unsuccessful response."""


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataProviderError(f"Request failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataProviderError(f"Unexpected response type from {url}")
    if payload.get("status") == "error" or payload.get("code"):
        raise DataProviderError(f"Provider error from {url}: {payload.get('message', payload)}")
    return payload


def fetch_twelve_data_bars(
    symbol: str, start: date, end: date, api_key: str
) -> list[dict[str, Any]]:
    """Fetch all daily bars for a symbol; values are returned oldest first."""
    payload = _get_json(
        TWELVE_DATA_URL,
        {
            "symbol": symbol,
            "interval": "1day",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "outputsize": 5000,
            "order": "ASC",
            "apikey": api_key,
        },
    )
    values = payload.get("values")
    if not isinstance(values, list):
        raise DataProviderError(f"Twelve Data returned no values for {symbol}: {payload}")
    bars = []
    for row in values:
        try:
            day = date.fromisoformat(row["datetime"][:10])
            if not start <= day <= end:
                continue
            bars.append(
                {
                    "bar_date": day,
                    "open": _optional_float(row.get("open")),
                    "high": _optional_float(row.get("high")),
                    "low": _optional_float(row.get("low")),
                    "close": float(row["close"]),
                    "volume": _optional_float(row.get("volume")),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataProviderError(f"Malformed Twelve Data bar for {symbol}: {row}") from exc
    return bars


def fetch_fred_tips(start: date, end: date, api_key: str) -> list[dict[str, Any]]:
    """Fetch DFII10 observations, omitting FRED's '.' missing-value markers."""
    payload = _get_json(
        FRED_OBSERVATIONS_URL,
        {
            "series_id": "DFII10",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "file_type": "json",
            "api_key": api_key,
        },
    )
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise DataProviderError(f"FRED returned no observations for DFII10: {payload}")
    result = []
    for item in observations:
        raw_value = item.get("value")
        if raw_value in (None, "."):
            continue
        try:
            observed_on = date.fromisoformat(item["date"])
            result.append({"observation_date": observed_on, "value": float(raw_value)})
        except (KeyError, TypeError, ValueError) as exc:
            raise DataProviderError(f"Malformed FRED DFII10 observation: {item}") from exc
    return result


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def _upsert_bars(session: Session, symbol: str, bars: list[dict[str, Any]]) -> int:
    for bar in bars:
        row = session.scalar(
            select(MarketBar).where(
                MarketBar.symbol == symbol, MarketBar.bar_date == bar["bar_date"]
            )
        )
        if row is None:
            row = MarketBar(symbol=symbol, bar_date=bar["bar_date"], provider="twelve_data")
            session.add(row)
        for field in ("open", "high", "low", "close", "volume"):
            setattr(row, field, bar[field])
    return len(bars)


def _upsert_tips(session: Session, observations: list[dict[str, Any]]) -> int:
    for observation in observations:
        row = session.scalar(
            select(EconomicObservation).where(
                EconomicObservation.series_id == "DFII10",
                EconomicObservation.observation_date == observation["observation_date"],
            )
        )
        if row is None:
            row = EconomicObservation(
                series_id="DFII10",
                observation_date=observation["observation_date"],
                unit="percent",
                provider="fred",
                value=observation["value"],
            )
            session.add(row)
        else:
            row.value = observation["value"]
    return len(observations)


def fetch_and_store(start: date, end: date) -> dict[str, int]:
    """Fetch all requested series and atomically upsert them into PostgreSQL."""
    database_url = os.environ.get("DATABASE_URL")
    twelve_key = os.environ.get("TWELVE_DATA_API_KEY")
    fred_key = os.environ.get("FRED_API_KEY")
    missing = [
        name
        for name, value in (
            ("DATABASE_URL", database_url),
            ("TWELVE_DATA_API_KEY", twelve_key),
            ("FRED_API_KEY", fred_key),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    if start > end:
        raise ValueError("start date must be on or before end date")

    # Pull data before opening the write transaction so provider failures cannot
    # leave a partially updated date range in the database.
    dxy_symbol = os.environ.get("TWELVE_DATA_DXY_SYMBOL", "DXY")
    gold_bars = fetch_twelve_data_bars("XAU/USD", start, end, twelve_key)
    dxy_bars = fetch_twelve_data_bars(dxy_symbol, start, end, twelve_key)
    tips = fetch_fred_tips(start, end, fred_key)

    engine = create_engine(database_url, pool_pre_ping=True)
    with Session(engine) as session, session.begin():
        counts = {
            "XAU/USD": _upsert_bars(session, "XAU/USD", gold_bars),
            "DXY": _upsert_bars(session, "DXY", dxy_bars),
            "DFII10": _upsert_tips(session, tips),
        }
    engine.dispose()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    try:
        counts = fetch_and_store(args.start_date, args.end_date)
    except (DataProviderError, ValueError) as exc:
        print(f"Data load failed: {exc}", file=sys.stderr)
        return 1
    for series, count in counts.items():
        print(f"{series}: upserted {count} observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
