"""add thread summary and user memory

Revision ID: 20260403_000006
Revises: 20260324_000005
Create Date: 2026-04-03 00:00:06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260403_000006"
down_revision = "20260324_000005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Add summary columns to chat_threads
    thread_columns = {col["name"] for col in inspector.get_columns("chat_threads")}
    if "summary" not in thread_columns:
        op.add_column(
            "chat_threads",
            sa.Column("summary", sa.Text(), nullable=True),
        )
    if "summary_message_count" not in thread_columns:
        op.add_column(
            "chat_threads",
            sa.Column(
                "summary_message_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        )

    # Create user_memories table
    existing_tables = inspector.get_table_names()
    if "user_memories" not in existing_tables:
        op.create_table(
            "user_memories",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("user_id", sa.String(36), nullable=False),
            sa.Column("key", sa.String(100), nullable=False),
            sa.Column("value", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "key", name="uq_user_memory_user_key"),
        )
        op.create_index("ix_user_memories_user_id", "user_memories", ["user_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    existing_tables = inspector.get_table_names()
    if "user_memories" in existing_tables:
        op.drop_index("ix_user_memories_user_id", table_name="user_memories")
        op.drop_table("user_memories")

    thread_columns = {col["name"] for col in inspector.get_columns("chat_threads")}
    if "summary_message_count" in thread_columns:
        op.drop_column("chat_threads", "summary_message_count")
    if "summary" in thread_columns:
        op.drop_column("chat_threads", "summary")
