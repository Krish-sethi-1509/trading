"""Add daily research and macro observation tables.

Revision ID: 0002_daily_research_tables
Revises: 0001_initial
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_daily_research_tables"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "market_bars" not in existing:
        op.create_table(
            "market_bars",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("symbol", sa.String(length=32), nullable=False),
            sa.Column("bar_date", sa.Date(), nullable=False),
            sa.Column("open", sa.Float(), nullable=True),
            sa.Column("high", sa.Float(), nullable=True),
            sa.Column("low", sa.Float(), nullable=True),
            sa.Column("close", sa.Float(), nullable=False),
            sa.Column("volume", sa.Float(), nullable=True),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.UniqueConstraint("symbol", "bar_date", name="uq_market_bars_symbol_date"),
        )
    existing_indexes = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("market_bars")} if "market_bars" in existing else set()
    if "ix_market_bars_symbol_date" not in existing_indexes:
        op.create_index("ix_market_bars_symbol_date", "market_bars", ["symbol", "bar_date"])

    if "economic_observations" not in existing:
        op.create_table(
            "economic_observations",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("series_id", sa.String(length=32), nullable=False),
            sa.Column("observation_date", sa.Date(), nullable=False),
            sa.Column("value", sa.Float(), nullable=False),
            sa.Column("unit", sa.String(length=32), nullable=False),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.UniqueConstraint("series_id", "observation_date", name="uq_economic_series_date"),
        )
    existing_indexes = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("economic_observations")} if "economic_observations" in existing else set()
    if "ix_economic_series_date" not in existing_indexes:
        op.create_index("ix_economic_series_date", "economic_observations", ["series_id", "observation_date"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "economic_observations" in tables:
        indexes = {index["name"] for index in inspector.get_indexes("economic_observations")}
        if "ix_economic_series_date" in indexes:
            op.drop_index("ix_economic_series_date", table_name="economic_observations")
        op.drop_table("economic_observations")
    if "market_bars" in tables:
        indexes = {index["name"] for index in inspector.get_indexes("market_bars")}
        if "ix_market_bars_symbol_date" in indexes:
            op.drop_index("ix_market_bars_symbol_date", table_name="market_bars")
        op.drop_table("market_bars")
