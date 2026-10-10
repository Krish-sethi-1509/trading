"""FastAPI application for the XAU/USD prediction MVP."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from database import get_db
from models import PredictionLog, PriceHistory
from services import (
    ServiceUnavailable,
    active_session,
    create_prediction,
    refresh_live_price,
    update_prediction_outcomes,
)
from chat_service import ChatServiceError, answer_query, search_market_sources
from market_service import build_market_snapshot

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
UTC = timezone.utc

HISTORY_INTERVAL_ROWS = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1_440,
    "1w": 10_080,
    "1mo": 43_200,
}


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=1200)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1200)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Schema changes are applied by Alembic before the API process starts.
    yield


limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
app = FastAPI(title="Gold Direction Prediction API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
origins = [
    value.strip()
    for value in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if value.strip()
]
frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
if frontend_url and frontend_url not in origins:
    origins.append(frontend_url)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
@limiter.limit("5/minute")
async def chat(request: Request, chat_request: ChatRequest) -> dict:
    """Retrieve current macro/news context and return a cited, guarded answer."""
    from starlette.concurrency import run_in_threadpool

    try:
        return await run_in_threadpool(
            answer_query,
            chat_request.query,
            [message.model_dump() for message in chat_request.history],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatServiceError as exc:
        logger.warning("Chat service unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/market-insights")
def market_insights(db: Session = Depends(get_db)) -> dict:
    """Return price-derived daily range, technical horizons, and 7-day levels."""
    rows = db.scalars(
        select(PriceHistory)
        .where(PriceHistory.symbol == "XAU/USD")
        .order_by(PriceHistory.timestamp.desc())
        .limit(10_080)
    ).all()
    rows.reverse()
    return build_market_snapshot(rows)


@app.get("/market-news")
async def market_news(kind: str = Query(default="news", pattern="^(news|events)$")) -> dict:
    """Return sourced news or macro calendar search results without LLM synthesis."""
    from dataclasses import asdict
    from starlette.concurrency import run_in_threadpool

    try:
        sources = await run_in_threadpool(search_market_sources, kind)
        return {"kind": kind, "sources": [asdict(source) for source in sources]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatServiceError as exc:
        logger.warning("Market %s search unavailable: %s", kind, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/price/live")
@limiter.limit("30/minute")
def live_price(request: Request) -> dict:
    """Fetch a fresh quote and return its observed bid/ask spread if available."""
    try:
        quote = refresh_live_price()
    except ServiceUnavailable as exc:
        logger.warning("Live quote unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    timestamp = quote["timestamp"]
    return {
        "symbol": "XAU/USD",
        "price": quote["price"],
        "bid": quote["bid"],
        "ask": quote["ask"],
        "spread": quote["spread"],
        "timestamp": timestamp.isoformat(),
        "active_session": active_session(timestamp),
        "source": quote.get("source", "unknown"),
        "is_stale": bool(quote.get("is_stale", False)),
        "persistence_warning": bool(quote.get("persistence_warning", False)),
    }


@app.get("/history")
def history(
    start: datetime | None = Query(default=None, description="Inclusive ISO 8601 timestamp"),
    end: datetime | None = Query(default=None, description="Exclusive ISO 8601 timestamp"),
    limit: int = Query(default=1000, ge=1, le=10000),
    interval: str = Query(default="1m", description="Candle interval: 1m, 3m, 5m, 1h, 2h, 4h, 1d, 1w, or 1mo"),
    db: Session = Depends(get_db),
) -> dict:
    """Return stored price observations aggregated into the requested interval."""
    if interval not in HISTORY_INTERVAL_ROWS:
        raise HTTPException(status_code=422, detail=f"Unsupported interval. Choose one of: {', '.join(HISTORY_INTERVAL_ROWS)}")
    statement = select(PriceHistory).where(PriceHistory.symbol == "XAU/USD")
    if start is not None:
        statement = statement.where(PriceHistory.timestamp >= _as_utc(start))
    if end is not None:
        statement = statement.where(PriceHistory.timestamp < _as_utc(end))
    # Fetch enough one-minute observations to build the requested number of
    # larger candles. The cap keeps local API requests bounded for long history.
    raw_limit = min(max(limit * HISTORY_INTERVAL_ROWS[interval], limit), 250_000)
    rows = db.scalars(statement.order_by(PriceHistory.timestamp.desc()).limit(raw_limit)).all()
    rows.reverse()
    candles = _aggregate_history(rows, interval)[-limit:]
    return {
        "symbol": "XAU/USD",
        "interval": interval,
        "count": len(candles),
        "candles": candles,
    }


def _history_bucket(timestamp: datetime, interval: str) -> datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    timestamp = timestamp.astimezone(UTC)
    if interval in {"1m", "3m", "5m", "1h", "2h", "4h"}:
        minutes = HISTORY_INTERVAL_ROWS[interval]
        seconds = int(timestamp.timestamp())
        bucket_seconds = minutes * 60
        return datetime.fromtimestamp((seconds // bucket_seconds) * bucket_seconds, tz=UTC)
    if interval == "1d":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if interval == "1w":
        start_of_week = timestamp.date() - timedelta(days=timestamp.weekday())
        return datetime(start_of_week.year, start_of_week.month, start_of_week.day, tzinfo=UTC)
    return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _aggregate_history(rows: list[PriceHistory], interval: str) -> list[dict]:
    """Aggregate chronological OHLC observations into UTC interval candles."""
    candles: dict[datetime, dict] = {}
    for row in rows:
        bucket = _history_bucket(row.timestamp, interval)
        candle = candles.get(bucket)
        if candle is None:
            candles[bucket] = {
                "timestamp": bucket.isoformat(),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "features": row.macro_features or {},
            }
            continue
        candle["high"] = max(candle["high"], row.high)
        candle["low"] = min(candle["low"], row.low)
        candle["close"] = row.close
        candle["features"] = row.macro_features or {}
        if row.volume is not None:
            candle["volume"] = (candle["volume"] or 0) + row.volume
    return [candles[key] for key in sorted(candles)]


@app.post("/predict")
@limiter.limit("10/minute")
def predict(request: Request, db: Session = Depends(get_db)) -> dict[str, str | float]:
    try:
        return create_prediction(db)
    except ServiceUnavailable as exc:
        logger.exception("Prediction could not be generated")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/accuracy-log")
def accuracy_log(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> dict:
    """Return running scored accuracy plus recent rows; scoring is scheduler-owned."""
    total, correct = db.execute(
        select(
            func.count(PredictionLog.id),
            func.coalesce(func.sum(PredictionLog.accuracy_flag.cast(Integer)), 0),
        ).where(PredictionLog.accuracy_flag.is_not(None))
    ).one()
    total, correct = int(total), int(correct)
    recent = db.scalars(
        select(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit)
    ).all()
    return {
        "status": "ready" if total else "no_evaluated_predictions",
        "accuracy_percent": (100.0 * correct / total) if total else None,
        "scored_predictions": total,
        "correct_predictions": correct,
        "pending_predictions": db.scalar(
            select(func.count(PredictionLog.id)).where(PredictionLog.accuracy_flag.is_(None))
        ),
        "recent": [
            {
                "timestamp": row.timestamp.isoformat(),
                "target_timestamp": row.target_timestamp.isoformat(),
                "predicted_direction": row.predicted_direction,
                "confidence": row.confidence_score,
                "reference_price": row.reference_price,
                "actual_outcome": row.actual_outcome,
                "correct": row.accuracy_flag,
            }
            for row in recent
        ],
    }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
