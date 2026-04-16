"""add is_admin to users

Revision ID: 20260415_000011
Revises: 20260415_000010
Create Date: 2026-04-15 00:00:11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260415_000011"
down_revision = "20260415_000010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {col["name"] for col in inspector.get_columns("users")}
    if "is_admin" not in existing:
        with op.batch_alter_table("users") as batch_op:
            batch_op.add_column(
                sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false())
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {col["name"] for col in inspector.get_columns("users")}
    if "is_admin" in existing:
        with op.batch_alter_table("users") as batch_op:
            batch_op.drop_column("is_admin")
