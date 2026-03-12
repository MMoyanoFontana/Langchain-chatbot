"""initial schema

Revision ID: 20260311_000001
Revises:
Create Date: 2026-03-11 00:00:01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260311_000001"
down_revision = None
branch_labels = None
depends_on = None

provider_code_enum = sa.Enum(
    "OPENAI",
    "GEMINI",
    "ANTHROPIC",
    "GROQ",
    "XAI",
    "OPENROUTER",
    "OTHER",
    name="provider_code",
    native_enum=False,
    create_constraint=False,
)
message_role_enum = sa.Enum(
    "SYSTEM",
    "USER",
    "ASSISTANT",
    "TOOL",
    name="message_role",
    native_enum=False,
    create_constraint=False,
)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "providers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", provider_code_enum, nullable=False),
        sa.Column("display_name", sa.String(length=80), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_providers_code", "providers", ["code"], unique=True)

    op.create_table(
        "provider_api_keys",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("key_name", sa.String(length=100), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "provider_id",
            "key_name",
            name="uq_provider_api_key_user_provider_name",
        ),
    )
    op.create_index("ix_provider_api_keys_provider_id", "provider_api_keys", ["provider_id"], unique=False)
    op.create_index("ix_provider_api_keys_user_id", "provider_api_keys", ["user_id"], unique=False)

    op.create_table(
        "provider_models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=120), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider_id",
            "model_id",
            name="uq_provider_model_provider_id_model_id",
        ),
    )
    op.create_index("ix_provider_models_provider_id", "provider_models", ["provider_id"], unique=False)

    op.create_table(
        "chat_threads",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("provider_api_key_id", sa.String(length=36), nullable=True),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["provider_api_key_id"], ["provider_api_keys.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_threads_provider_api_key_id",
        "chat_threads",
        ["provider_api_key_id"],
        unique=False,
    )
    op.create_index("ix_chat_threads_user_id", "chat_threads", ["user_id"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=False),
        sa.Column("role", message_role_enum, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(length=120), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["thread_id"], ["chat_threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_provider_id", "chat_messages", ["provider_id"], unique=False)
    op.create_index("ix_chat_messages_role", "chat_messages", ["role"], unique=False)
    op.create_index("ix_chat_messages_thread_id", "chat_messages", ["thread_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_chat_messages_thread_id", table_name="chat_messages")
    op.drop_index("ix_chat_messages_role", table_name="chat_messages")
    op.drop_index("ix_chat_messages_provider_id", table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index("ix_chat_threads_user_id", table_name="chat_threads")
    op.drop_index("ix_chat_threads_provider_api_key_id", table_name="chat_threads")
    op.drop_table("chat_threads")

    op.drop_index("ix_provider_models_provider_id", table_name="provider_models")
    op.drop_table("provider_models")

    op.drop_index("ix_provider_api_keys_user_id", table_name="provider_api_keys")
    op.drop_index("ix_provider_api_keys_provider_id", table_name="provider_api_keys")
    op.drop_table("provider_api_keys")

    op.drop_index("ix_providers_code", table_name="providers")
    op.drop_table("providers")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
