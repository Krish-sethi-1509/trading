"""Create prediction and price history tables.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-09
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "price_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=24), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("macro_features", sa.JSON(), nullable=False),
        sa.Column("feature_vector", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_price_history_symbol_timestamp"),
    )
    op.create_index("ix_price_history_timestamp", "price_history", ["timestamp"], unique=False)

    op.create_table(
        "predictions_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("target_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_direction", sa.String(length=8), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("reference_price", sa.Float(), nullable=False),
        sa.Column("actual_outcome", sa.String(length=8), nullable=True),
        sa.Column("accuracy_flag", sa.Boolean(), nullable=True),
        sa.Column("actualized_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_predictions_log_timestamp", "predictions_log", ["timestamp"], unique=False)
    op.create_index("ix_predictions_log_actualized", "predictions_log", ["actualized_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_predictions_log_actualized", table_name="predictions_log")
    op.drop_index("ix_predictions_log_timestamp", table_name="predictions_log")
    op.drop_table("predictions_log")
    op.drop_index("ix_price_history_timestamp", table_name="price_history")
    op.drop_table("price_history")
