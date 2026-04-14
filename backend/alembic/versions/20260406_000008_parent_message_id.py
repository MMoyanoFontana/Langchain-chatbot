"""add parent_message_id to chat_messages

Revision ID: 20260406_000008
Revises: 20260403_000007
Create Date: 2026-04-06 00:00:08
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260406_000008"
down_revision = "20260403_000007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    chat_message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "parent_message_id" not in chat_message_columns:
        with op.batch_alter_table("chat_messages") as batch_op:
            batch_op.add_column(
                sa.Column("parent_message_id", sa.String(36), nullable=True),
            )
            batch_op.create_index(
                "ix_chat_messages_parent_message_id", ["parent_message_id"]
            )
            batch_op.create_foreign_key(
                "fk_chat_messages_parent_message_id",
                "chat_messages",
                ["parent_message_id"],
                ["id"],
                ondelete="SET NULL",
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    chat_message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "parent_message_id" in chat_message_columns:
        op.drop_column("chat_messages", "parent_message_id")
