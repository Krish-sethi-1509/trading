"""Derive transparent market snapshot metrics from persisted XAU/USD observations."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

UTC = timezone.utc
HORIZONS = (
    ("Short-term", 60),
    ("Intermediate", 240),
    ("Long-term", 1_440),
)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _price_return_and_score(rows: Sequence[Any], minutes: int) -> tuple[float | None, float | None, str]:
    if len(rows) < 2:
        return None, None, "INSUFFICIENT_DATA"
    newest_time = _as_utc(rows[-1].timestamp)
    target_time = newest_time - timedelta(minutes=minutes)
    if _as_utc(rows[0].timestamp) > target_time:
        return None, None, "INSUFFICIENT_DATA"
    start_index = next(
        (index for index, row in enumerate(rows) if _as_utc(row.timestamp) >= target_time),
        len(rows),
    )
    if start_index >= len(rows) or start_index == len(rows) - 1:
        return None, None, "INSUFFICIENT_DATA"
    start_price = float(rows[start_index].close)
    end_price = float(rows[-1].close)
    if start_price <= 0 or end_price <= 0:
        return None, None, "INSUFFICIENT_DATA"
    change_pct = (end_price / start_price - 1.0) * 100.0
    returns = []
    for previous, current in zip(rows[start_index:-1], rows[start_index + 1 :]):
        previous_close = float(previous.close)
        current_close = float(current.close)
        if previous_close > 0 and current_close > 0:
            returns.append((current_close / previous_close - 1.0) * 100.0)
    cumulative_volatility = math.sqrt(sum(value * value for value in returns))
    normalized_move = change_pct / cumulative_volatility if cumulative_volatility > 1e-12 else 0.0
    score = max(0.0, min(100.0, 50.0 + 25.0 * max(-2.0, min(2.0, normalized_move))))
    direction = "BULLISH" if score >= 55 else "BEARISH" if score <= 45 else "NEUTRAL"
    return change_pct, score, direction


def build_market_snapshot(rows: Sequence[Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Calculate daily range, multi-horizon technical scores, and rolling levels.

    The input consists of persisted observations. When an upstream quote API
    supplies only point prices, these values describe observed quote samples,
    not exchange-traded OHLCV candles or volume-weighted sentiment.
    """
    current_time = _as_utc(now or datetime.now(UTC))
    ordered = sorted(rows, key=lambda row: _as_utc(row.timestamp))[-10_080:]
    if not ordered:
        return {
            "status": "no_data",
            "as_of": current_time.isoformat(),
            "price": None,
            "daily_change_pct": None,
            "day_high": None,
            "day_low": None,
            "range_position_pct": None,
            "technical_scores": [
                {"label": label, "minutes": minutes, "score": None, "direction": "INSUFFICIENT_DATA", "change_pct": None}
                for label, minutes in HORIZONS
            ],
            "support": None,
            "resistance": None,
            "sentiment": None,
            "overview": "Market observations are not available yet. Start the backend price refresh and try again.",
            "data_note": "Metrics use stored price observations; actual traded volume is not available from the configured quote feed.",
        }

    normalized_times = [_as_utc(row.timestamp) for row in ordered]
    newest_time = normalized_times[-1]
    current_price = float(ordered[-1].close)
    utc_day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    day_rows = [row for row, stamp in zip(ordered, normalized_times) if utc_day_start <= stamp <= current_time]
    daily_change_pct = None
    day_high = day_low = range_position_pct = None
    if day_rows:
        first_price = float(day_rows[0].open)
        day_high = max(float(row.high) for row in day_rows)
        day_low = min(float(row.low) for row in day_rows)
        if first_price > 0:
            daily_change_pct = (current_price / first_price - 1.0) * 100.0
        span = day_high - day_low
        if span > 0:
            range_position_pct = max(0.0, min(100.0, (current_price - day_low) / span * 100.0))

    technical_scores = []
    for label, minutes in HORIZONS:
        change_pct, score, direction = _price_return_and_score(ordered, minutes)
        technical_scores.append({
            "label": label,
            "minutes": minutes,
            "score": round(score, 1) if score is not None else None,
            "direction": direction,
            "change_pct": round(change_pct, 4) if change_pct is not None else None,
        })

    lookback_start = newest_time - timedelta(days=7)
    lookback_rows = [row for row, stamp in zip(ordered, normalized_times) if stamp >= lookback_start]
    lower_levels = [float(row.low) for row in lookback_rows if float(row.low) < current_price]
    upper_levels = [float(row.high) for row in lookback_rows if float(row.high) > current_price]
    support = max(lower_levels) if lower_levels else None
    resistance = min(upper_levels) if upper_levels else None

    if daily_change_pct is None:
        overview = "There are no stored observations for the current UTC day yet."
    else:
        move_word = "rose" if daily_change_pct > 0 else "fell" if daily_change_pct < 0 else "was little changed"
        daily_phrase = f"Gold {move_word} {abs(daily_change_pct):.2f}% from the first stored quote today"
        active_signals = [item for item in technical_scores if item["score"] is not None]
        strongest = active_signals[-1] if active_signals else None
        if strongest:
            overview = f"{daily_phrase}. The {strongest['label'].lower()} technical score is {strongest['direction'].lower()} ({strongest['score']:.0f}/100)."
        else:
            overview = f"{daily_phrase}. More observations are needed to calculate technical scores."

    return {
        "status": "ready",
        "as_of": newest_time.isoformat(),
        "price": current_price,
        "daily_change_pct": round(daily_change_pct, 4) if daily_change_pct is not None else None,
        "day_high": day_high,
        "day_low": day_low,
        "range_position_pct": round(range_position_pct, 1) if range_position_pct is not None else None,
        "technical_scores": technical_scores,
        "support": support,
        "resistance": resistance,
        "sentiment": None,
        "overview": overview,
        "data_note": "Daily range and technical scores use stored quote samples in UTC. Support/resistance are observed 7-day levels. The configured feed does not provide traded volume or broker positioning.",
    }
