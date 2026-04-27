"""add username and restore password_hash

Revision ID: 20260427_000014
Revises: 20260427_000013
Create Date: 2026-04-27 00:00:14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260427_000014"
down_revision = "20260427_000013"
branch_labels = None
depends_on = None

_ORPHAN_INDEXES = [
    "ix_users_email_verification_token",
    "ix_users_email_verification_expires_at",
    "ix_users_email_verified",
]
_ORPHAN_COLS = ["email_verified", "email_verification_token", "email_verification_expires_at"]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {col["name"] for col in inspector.get_columns("users")}
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("users")}

    # Drop orphaned indexes before batch so recreate doesn't try to rebuild them
    for idx in _ORPHAN_INDEXES:
        if idx in existing_indexes:
            op.drop_index(idx, table_name="users")

    with op.batch_alter_table("users", recreate="always") as batch_op:
        for col in _ORPHAN_COLS:
            if col in existing_cols:
                batch_op.drop_column(col)
        if "username" not in existing_cols:
            batch_op.add_column(sa.Column("username", sa.String(30), nullable=True))
        if "password_hash" not in existing_cols:
            batch_op.add_column(sa.Column("password_hash", sa.String(255), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("users", recreate="always") as batch_op:
        batch_op.drop_column("username")
        batch_op.drop_column("password_hash")
