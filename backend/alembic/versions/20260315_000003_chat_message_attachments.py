"""add chat message attachments

Revision ID: 20260315_000003
Revises: 20260311_000002
Create Date: 2026-03-15 00:00:03
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260315_000003"
down_revision = "20260311_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    chat_message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "attachments" not in chat_message_columns:
        op.add_column(
            "chat_messages",
            sa.Column(
                "attachments",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    chat_message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "attachments" in chat_message_columns:
        op.drop_column("chat_messages", "attachments")
