"""add authentication schema

Revision ID: 20260311_000002
Revises: 20260311_000001
Create Date: 2026-03-11 00:00:02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260311_000002"
down_revision = "20260311_000001"
branch_labels = None
depends_on = None

auth_provider_enum = sa.Enum(
    "GOOGLE",
    "GITHUB",
    "MICROSOFT",
    name="auth_provider",
    native_enum=False,
    create_constraint=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {column["name"] for column in inspector.get_columns("users")}
    if "password_hash" not in user_columns:
        op.add_column("users", sa.Column("password_hash", sa.String(length=255), nullable=True))
    if "avatar_url" not in user_columns:
        op.add_column("users", sa.Column("avatar_url", sa.String(length=2048), nullable=True))

    table_names = set(inspector.get_table_names())
    if "auth_identities" not in table_names:
        op.create_table(
            "auth_identities",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("provider", auth_provider_enum, nullable=False),
            sa.Column("provider_subject", sa.String(length=255), nullable=False),
            sa.Column("email", sa.String(length=320), nullable=True),
            sa.Column("avatar_url", sa.String(length=2048), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "provider",
                "provider_subject",
                name="uq_auth_identity_provider_subject",
            ),
        )

    identity_indexes = {index["name"] for index in sa.inspect(bind).get_indexes("auth_identities")}
    if "ix_auth_identities_provider" not in identity_indexes:
        op.create_index("ix_auth_identities_provider", "auth_identities", ["provider"], unique=False)
    if "ix_auth_identities_user_id" not in identity_indexes:
        op.create_index("ix_auth_identities_user_id", "auth_identities", ["user_id"], unique=False)

    if "auth_sessions" not in table_names:
        op.create_table(
            "auth_sessions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    session_indexes = {index["name"] for index in sa.inspect(bind).get_indexes("auth_sessions")}
    if "ix_auth_sessions_token_hash" not in session_indexes:
        op.create_index("ix_auth_sessions_token_hash", "auth_sessions", ["token_hash"], unique=False)
    if "ix_auth_sessions_user_id" not in session_indexes:
        op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())

    if "auth_sessions" in table_names:
        session_indexes = {index["name"] for index in inspector.get_indexes("auth_sessions")}
        if "ix_auth_sessions_user_id" in session_indexes:
            op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
        if "ix_auth_sessions_token_hash" in session_indexes:
            op.drop_index("ix_auth_sessions_token_hash", table_name="auth_sessions")
        op.drop_table("auth_sessions")

    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "auth_identities" in table_names:
        identity_indexes = {index["name"] for index in inspector.get_indexes("auth_identities")}
        if "ix_auth_identities_user_id" in identity_indexes:
            op.drop_index("ix_auth_identities_user_id", table_name="auth_identities")
        if "ix_auth_identities_provider" in identity_indexes:
            op.drop_index("ix_auth_identities_provider", table_name="auth_identities")
        op.drop_table("auth_identities")

    user_columns = {column["name"] for column in sa.inspect(bind).get_columns("users")}
    if "avatar_url" in user_columns:
        op.drop_column("users", "avatar_url")
    if "password_hash" in user_columns:
        op.drop_column("users", "password_hash")
