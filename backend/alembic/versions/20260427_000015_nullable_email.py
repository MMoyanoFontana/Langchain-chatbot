"""make users.email nullable, clear placeholder emails

Revision ID: 20260427_000015
Revises: 20260427_000014
Create Date: 2026-04-27 00:00:15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260427_000015"
down_revision = "20260427_000014"
branch_labels = None
depends_on = None

PLACEHOLDER_DOMAIN = "users.local"


def upgrade() -> None:
    # Make column nullable first, then clear placeholder emails
    with op.batch_alter_table("users", recreate="always") as batch_op:
        batch_op.alter_column("email", existing_type=sa.String(320), nullable=True)

    bind = op.get_bind()
    bind.execute(
        sa.text("UPDATE users SET email = NULL WHERE email LIKE :pattern AND username IS NOT NULL"),
        {"pattern": f"%@{PLACEHOLDER_DOMAIN}"},
    )


def downgrade() -> None:
    # Restore placeholder emails before making column NOT NULL again
    bind = op.get_bind()
    bind.execute(
        sa.text("UPDATE users SET email = username || :suffix WHERE email IS NULL AND username IS NOT NULL"),
        {"suffix": f"@{PLACEHOLDER_DOMAIN}"},
    )

    with op.batch_alter_table("users", recreate="always") as batch_op:
        batch_op.alter_column("email", existing_type=sa.String(320), nullable=False)
