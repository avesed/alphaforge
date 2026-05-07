"""Prediction, model, and factor tables for local ML storage.

Revision ID: 002_prediction_tables
Revises: 001_initial_schema
Create Date: 2026-05-06
"""

from alembic import op
import sqlalchemy as sa

revision = "002_prediction_tables"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "prediction_models",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("market", sa.String(10), nullable=False),
        sa.Column("model_date", sa.Date, nullable=False),
        sa.Column("model_type", sa.String(20), nullable=False, server_default="lgbm_rank"),
        sa.Column("forward_days", sa.Integer, nullable=False, server_default=sa.text("5")),
        sa.Column("ic", sa.Float, nullable=True),
        sa.Column("icir", sa.Float, nullable=True),
        sa.Column("ndcg", sa.Float, nullable=True),
        sa.Column("fold_ics", sa.JSON, nullable=True),
        sa.Column("best_iterations", sa.JSON, nullable=True),
        sa.Column("feature_count", sa.Integer, nullable=True),
        sa.Column("symbol_count", sa.Integer, nullable=True),
        sa.Column("ensemble_size", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("quality", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("training_config", sa.JSON, nullable=True),
        sa.Column("feature_importance", sa.JSON, nullable=True),
        sa.Column("file_path", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_prediction_models_market", "prediction_models", ["market"])

    op.create_table(
        "stock_predictions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market", sa.String(10), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("prediction_date", sa.Date, nullable=False),
        sa.Column("forward_days", sa.Integer, nullable=False, server_default=sa.text("5")),
        sa.Column("rank_score", sa.Float, nullable=True),
        sa.Column("percentile_rank", sa.Float, nullable=True),
        sa.Column("up_probability", sa.Float, nullable=True),
        sa.Column("actual_return", sa.Float, nullable=True),
        sa.Column("model_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_stock_predictions_market", "stock_predictions", ["market"])
    op.create_index("ix_stock_predictions_symbol", "stock_predictions", ["symbol"])
    op.create_index("ix_stock_predictions_prediction_date", "stock_predictions", ["prediction_date"])
    op.create_index("ix_stock_predictions_model_id", "stock_predictions", ["model_id"])
    op.create_index("ix_pred_market_date", "stock_predictions", ["market", "prediction_date"])
    op.create_unique_constraint(
        "uq_prediction",
        "stock_predictions",
        ["market", "symbol", "prediction_date", "forward_days"],
    )

    op.create_table(
        "discovered_factors",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("market", sa.String(10), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("expression", sa.Text, nullable=False),
        sa.Column("ic", sa.Float, nullable=True),
        sa.Column("icir", sa.Float, nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="rdagent"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_discovered_factors_market", "discovered_factors", ["market"])


def downgrade():
    op.drop_table("discovered_factors")
    op.drop_table("stock_predictions")
    op.drop_table("prediction_models")
