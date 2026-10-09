"""PostgreSQL persistence models for chart data and prediction outcomes."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PriceHistory(Base):
    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_price_history_symbol_timestamp"),
        Index("ix_price_history_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False, default="XAU/USD")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float | None] = mapped_column(Float)
    # Feature columns evolve as the research pipeline changes; keep the
    # feature snapshot as JSON while retaining OHLCV as typed chart columns.
    macro_features: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    feature_vector: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class PredictionLog(Base):
    __tablename__ = "predictions_log"
    __table_args__ = (
        Index("ix_predictions_log_timestamp", "timestamp"),
        Index("ix_predictions_log_actualized", "actualized_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    target_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predicted_direction: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    reference_price: Mapped[float] = mapped_column(Float, nullable=False)
    actual_outcome: Mapped[str | None] = mapped_column(String(8))
    accuracy_flag: Mapped[bool | None] = mapped_column(Boolean)
    actualized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
