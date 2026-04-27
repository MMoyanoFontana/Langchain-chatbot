"""drop password_hash from users

Revision ID: 20260427_000013
Revises: 20260417_000012
Create Date: 2026-04-27 00:00:13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260427_000013"
down_revision = "20260417_000012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("password_hash")


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("password_hash", sa.String(255), nullable=True))
