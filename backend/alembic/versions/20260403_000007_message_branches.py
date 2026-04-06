"""add branch_index to chat_messages

Revision ID: 20260403_000007
Revises: 20260403_000006
Create Date: 2026-04-03 00:00:07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260403_000007"
down_revision = "20260403_000006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    message_columns = {col["name"] for col in inspector.get_columns("chat_messages")}
    if "branch_index" not in message_columns:
        op.add_column(
            "chat_messages",
            sa.Column(
                "branch_index",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    message_columns = {col["name"] for col in inspector.get_columns("chat_messages")}
    if "branch_index" in message_columns:
        op.drop_column("chat_messages", "branch_index")
