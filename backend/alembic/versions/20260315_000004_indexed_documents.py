"""add indexed documents for rag

Revision ID: 20260315_000004
Revises: 20260315_000003
Create Date: 2026-03-15 00:00:04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260315_000004"
down_revision = "20260315_000003"
branch_labels = None
depends_on = None

document_index_status_enum = sa.Enum(
    "PENDING",
    "INDEXED",
    "FAILED",
    name="document_index_status",
    native_enum=False,
    create_constraint=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())

    if "indexed_documents" not in table_names:
        op.create_table(
            "indexed_documents",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("thread_id", sa.String(length=36), nullable=False),
            sa.Column("source_message_id", sa.String(length=36), nullable=True),
            sa.Column("filename", sa.String(length=512), nullable=True),
            sa.Column("media_type", sa.String(length=255), nullable=False),
            sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
            sa.Column("byte_size", sa.Integer(), nullable=False),
            sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("pinecone_namespace", sa.String(length=255), nullable=False),
            sa.Column(
                "status",
                document_index_status_enum,
                nullable=False,
                server_default=sa.text("'PENDING'"),
            ),
            sa.Column("error_message", sa.String(length=500), nullable=True),
            sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["source_message_id"], ["chat_messages.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["thread_id"], ["chat_threads.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id",
                "thread_id",
                "checksum_sha256",
                name="uq_indexed_document_user_thread_checksum",
            ),
        )

    index_names = {index["name"] for index in sa.inspect(bind).get_indexes("indexed_documents")}
    if "ix_indexed_documents_user_id" not in index_names:
        op.create_index("ix_indexed_documents_user_id", "indexed_documents", ["user_id"], unique=False)
    if "ix_indexed_documents_thread_id" not in index_names:
        op.create_index("ix_indexed_documents_thread_id", "indexed_documents", ["thread_id"], unique=False)
    if "ix_indexed_documents_source_message_id" not in index_names:
        op.create_index(
            "ix_indexed_documents_source_message_id",
            "indexed_documents",
            ["source_message_id"],
            unique=False,
        )
    if "ix_indexed_documents_status" not in index_names:
        op.create_index("ix_indexed_documents_status", "indexed_documents", ["status"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "indexed_documents" not in table_names:
        return

    index_names = {index["name"] for index in inspector.get_indexes("indexed_documents")}
    if "ix_indexed_documents_status" in index_names:
        op.drop_index("ix_indexed_documents_status", table_name="indexed_documents")
    if "ix_indexed_documents_source_message_id" in index_names:
        op.drop_index("ix_indexed_documents_source_message_id", table_name="indexed_documents")
    if "ix_indexed_documents_thread_id" in index_names:
        op.drop_index("ix_indexed_documents_thread_id", table_name="indexed_documents")
    if "ix_indexed_documents_user_id" in index_names:
        op.drop_index("ix_indexed_documents_user_id", table_name="indexed_documents")
    op.drop_table("indexed_documents")
