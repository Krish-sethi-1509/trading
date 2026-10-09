"""SQLAlchemy models for daily XAU/USD, DXY, and US 10-year TIPS yields."""

from datetime import date

from sqlalchemy import Date, Float, Index, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for application ORM models."""


class MarketBar(Base):
    """Daily OHLCV market bar; volume may be unavailable for OTC spot feeds."""

    __tablename__ = "market_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "bar_date", name="uq_market_bars_symbol_date"),
        Index("ix_market_bars_symbol_date", "symbol", "bar_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    bar_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float | None] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)


class EconomicObservation(Base):
    """Daily macroeconomic observation, stored in the provider's published units."""

    __tablename__ = "economic_observations"
    __table_args__ = (
        UniqueConstraint("series_id", "observation_date", name="uq_economic_series_date"),
        Index("ix_economic_series_date", "series_id", "observation_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    series_id: Mapped[str] = mapped_column(String(32), nullable=False)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
