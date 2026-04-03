"""add chat message citations

Revision ID: 20260324_000005
Revises: 20260315_000004
Create Date: 2026-03-24 00:00:05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260324_000005"
down_revision = "20260315_000004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    chat_message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "citations" not in chat_message_columns:
        op.add_column(
            "chat_messages",
            sa.Column(
                "citations",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    chat_message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "citations" in chat_message_columns:
        op.drop_column("chat_messages", "citations")
