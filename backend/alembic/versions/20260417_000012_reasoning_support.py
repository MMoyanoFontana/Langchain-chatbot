"""add reasoning support

Revision ID: 20260417_000012
Revises: 20260415_000011
Create Date: 2026-04-17 00:00:12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260417_000012"
down_revision = "20260415_000011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    message_columns = {col["name"] for col in inspector.get_columns("chat_messages")}
    if "reasoning_content" not in message_columns:
        with op.batch_alter_table("chat_messages") as batch_op:
            batch_op.add_column(sa.Column("reasoning_content", sa.Text(), nullable=True))

    model_columns = {col["name"] for col in inspector.get_columns("provider_models")}
    if "supports_reasoning" not in model_columns:
        with op.batch_alter_table("provider_models") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "supports_reasoning",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                )
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    model_columns = {col["name"] for col in inspector.get_columns("provider_models")}
    if "supports_reasoning" in model_columns:
        with op.batch_alter_table("provider_models") as batch_op:
            batch_op.drop_column("supports_reasoning")

    message_columns = {col["name"] for col in inspector.get_columns("chat_messages")}
    if "reasoning_content" in message_columns:
        with op.batch_alter_table("chat_messages") as batch_op:
            batch_op.drop_column("reasoning_content")
