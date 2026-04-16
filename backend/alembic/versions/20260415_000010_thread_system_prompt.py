"""add system_prompt to chat_threads

Revision ID: 20260415_000010
Revises: 20260408_000009
Create Date: 2026-04-15 00:00:10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260415_000010"
down_revision = "20260408_000009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {col["name"] for col in inspector.get_columns("chat_threads")}
    if "system_prompt" not in existing:
        with op.batch_alter_table("chat_threads") as batch_op:
            batch_op.add_column(sa.Column("system_prompt", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {col["name"] for col in inspector.get_columns("chat_threads")}
    if "system_prompt" in existing:
        with op.batch_alter_table("chat_threads") as batch_op:
            batch_op.drop_column("system_prompt")
