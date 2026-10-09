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
    op.create_index("ix_market_bars_symbol_date", "market_bars", ["symbol", "bar_date"])
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
    op.create_index("ix_economic_series_date", "economic_observations", ["series_id", "observation_date"])


def downgrade() -> None:
    op.drop_index("ix_economic_series_date", table_name="economic_observations")
    op.drop_table("economic_observations")
    op.drop_index("ix_market_bars_symbol_date", table_name="market_bars")
    op.drop_table("market_bars")
