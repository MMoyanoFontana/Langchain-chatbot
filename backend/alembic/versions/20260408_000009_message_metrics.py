"""add latency and total_tokens metrics to chat_messages

Revision ID: 20260408_000009
Revises: 20260406_000008
Create Date: 2026-04-08 00:00:09
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260408_000009"
down_revision = "20260406_000008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {col["name"] for col in inspector.get_columns("chat_messages")}

    to_add = [
        ("total_tokens", sa.Integer()),
        ("latency_ms", sa.Integer()),
        ("time_to_first_token_ms", sa.Integer()),
    ]
    missing = [(name, type_) for name, type_ in to_add if name not in existing]
    if not missing:
        return

    with op.batch_alter_table("chat_messages") as batch_op:
        for name, type_ in missing:
            batch_op.add_column(sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {col["name"] for col in inspector.get_columns("chat_messages")}
    for name in ("total_tokens", "latency_ms", "time_to_first_token_ms"):
        if name in existing:
            op.drop_column("chat_messages", name)
