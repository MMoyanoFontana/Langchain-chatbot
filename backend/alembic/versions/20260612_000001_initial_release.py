"""initial release

Revision ID: 20260612_000001
Revises: 
Create Date: 2026-06-12 05:32:37.138613
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '20260612_000001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('providers',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('code', sa.Enum('OPENAI', 'GEMINI', 'ANTHROPIC', 'GROQ', 'OLLAMA', 'OTHER', name='provider_code', native_enum=False), nullable=False),
    sa.Column('display_name', sa.String(length=80), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_providers_code'), 'providers', ['code'], unique=True)
    op.create_table('users',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=True),
    sa.Column('username', sa.String(length=30), nullable=True),
    sa.Column('full_name', sa.String(length=120), nullable=True),
    sa.Column('password_hash', sa.String(length=255), nullable=True),
    sa.Column('avatar_url', sa.String(length=2048), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('is_admin', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_table('auth_identities',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('provider', sa.Enum('GOOGLE', 'GITHUB', 'MICROSOFT', name='auth_provider', native_enum=False), nullable=False),
    sa.Column('provider_subject', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=True),
    sa.Column('avatar_url', sa.String(length=2048), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('provider', 'provider_subject', name='uq_auth_identity_provider_subject')
    )
    op.create_index(op.f('ix_auth_identities_provider'), 'auth_identities', ['provider'], unique=False)
    op.create_index(op.f('ix_auth_identities_user_id'), 'auth_identities', ['user_id'], unique=False)
    op.create_table('auth_sessions',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_auth_sessions_token_hash'), 'auth_sessions', ['token_hash'], unique=False)
    op.create_index(op.f('ix_auth_sessions_user_id'), 'auth_sessions', ['user_id'], unique=False)
    op.create_table('provider_api_keys',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('provider_id', sa.Integer(), nullable=False),
    sa.Column('key_name', sa.String(length=100), nullable=False),
    sa.Column('encrypted_api_key', sa.Text(), nullable=False),
    sa.Column('is_default', sa.Boolean(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['provider_id'], ['providers.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'provider_id', 'key_name', name='uq_provider_api_key_user_provider_name')
    )
    op.create_index(op.f('ix_provider_api_keys_provider_id'), 'provider_api_keys', ['provider_id'], unique=False)
    op.create_index(op.f('ix_provider_api_keys_user_id'), 'provider_api_keys', ['user_id'], unique=False)
    op.create_table('provider_models',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('provider_id', sa.Integer(), nullable=False),
    sa.Column('model_id', sa.String(length=120), nullable=False),
    sa.Column('display_name', sa.String(length=120), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('supports_reasoning', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['provider_id'], ['providers.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('provider_id', 'model_id', name='uq_provider_model_provider_id_model_id')
    )
    op.create_index(op.f('ix_provider_models_provider_id'), 'provider_models', ['provider_id'], unique=False)
    op.create_table('user_memories',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('key', sa.String(length=100), nullable=False),
    sa.Column('value', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'key', name='uq_user_memory_user_key')
    )
    op.create_index(op.f('ix_user_memories_user_id'), 'user_memories', ['user_id'], unique=False)
    op.create_table('chat_threads',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('provider_api_key_id', sa.String(length=36), nullable=True),
    sa.Column('title', sa.String(length=200), nullable=True),
    sa.Column('system_prompt', sa.Text(), nullable=True),
    sa.Column('summary', sa.Text(), nullable=True),
    sa.Column('summary_message_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['provider_api_key_id'], ['provider_api_keys.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_threads_provider_api_key_id'), 'chat_threads', ['provider_api_key_id'], unique=False)
    op.create_index(op.f('ix_chat_threads_user_id'), 'chat_threads', ['user_id'], unique=False)
    op.create_table('chat_messages',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('thread_id', sa.String(length=36), nullable=False),
    sa.Column('role', sa.Enum('SYSTEM', 'USER', 'ASSISTANT', 'TOOL', name='message_role', native_enum=False), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('reasoning_content', sa.Text(), nullable=True),
    sa.Column('attachments', sa.JSON(), nullable=False),
    sa.Column('citations', sa.JSON(), nullable=False),
    sa.Column('provider_id', sa.Integer(), nullable=True),
    sa.Column('model_name', sa.String(length=120), nullable=True),
    sa.Column('parent_message_id', sa.String(length=36), nullable=True),
    sa.Column('branch_index', sa.Integer(), nullable=False),
    sa.Column('prompt_tokens', sa.Integer(), nullable=True),
    sa.Column('completion_tokens', sa.Integer(), nullable=True),
    sa.Column('total_tokens', sa.Integer(), nullable=True),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('time_to_first_token_ms', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['parent_message_id'], ['chat_messages.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['provider_id'], ['providers.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['thread_id'], ['chat_threads.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_messages_parent_message_id'), 'chat_messages', ['parent_message_id'], unique=False)
    op.create_index(op.f('ix_chat_messages_provider_id'), 'chat_messages', ['provider_id'], unique=False)
    op.create_index(op.f('ix_chat_messages_role'), 'chat_messages', ['role'], unique=False)
    op.create_index(op.f('ix_chat_messages_thread_id'), 'chat_messages', ['thread_id'], unique=False)
    op.create_table('indexed_documents',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('thread_id', sa.String(length=36), nullable=False),
    sa.Column('source_message_id', sa.String(length=36), nullable=True),
    sa.Column('filename', sa.String(length=512), nullable=True),
    sa.Column('media_type', sa.String(length=255), nullable=False),
    sa.Column('checksum_sha256', sa.String(length=64), nullable=False),
    sa.Column('byte_size', sa.Integer(), nullable=False),
    sa.Column('chunk_count', sa.Integer(), nullable=False),
    sa.Column('pinecone_namespace', sa.String(length=255), nullable=False),
    sa.Column('status', sa.Enum('PENDING', 'INDEXED', 'FAILED', name='document_index_status', native_enum=False), nullable=False),
    sa.Column('error_message', sa.String(length=500), nullable=True),
    sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_message_id'], ['chat_messages.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['thread_id'], ['chat_threads.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'thread_id', 'checksum_sha256', name='uq_indexed_document_user_thread_checksum')
    )
    op.create_index(op.f('ix_indexed_documents_source_message_id'), 'indexed_documents', ['source_message_id'], unique=False)
    op.create_index(op.f('ix_indexed_documents_status'), 'indexed_documents', ['status'], unique=False)
    op.create_index(op.f('ix_indexed_documents_thread_id'), 'indexed_documents', ['thread_id'], unique=False)
    op.create_index(op.f('ix_indexed_documents_user_id'), 'indexed_documents', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_indexed_documents_user_id'), table_name='indexed_documents')
    op.drop_index(op.f('ix_indexed_documents_thread_id'), table_name='indexed_documents')
    op.drop_index(op.f('ix_indexed_documents_status'), table_name='indexed_documents')
    op.drop_index(op.f('ix_indexed_documents_source_message_id'), table_name='indexed_documents')
    op.drop_table('indexed_documents')
    op.drop_index(op.f('ix_chat_messages_thread_id'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_role'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_provider_id'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_parent_message_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
    op.drop_index(op.f('ix_chat_threads_user_id'), table_name='chat_threads')
    op.drop_index(op.f('ix_chat_threads_provider_api_key_id'), table_name='chat_threads')
    op.drop_table('chat_threads')
    op.drop_index(op.f('ix_user_memories_user_id'), table_name='user_memories')
    op.drop_table('user_memories')
    op.drop_index(op.f('ix_provider_models_provider_id'), table_name='provider_models')
    op.drop_table('provider_models')
    op.drop_index(op.f('ix_provider_api_keys_user_id'), table_name='provider_api_keys')
    op.drop_index(op.f('ix_provider_api_keys_provider_id'), table_name='provider_api_keys')
    op.drop_table('provider_api_keys')
    op.drop_index(op.f('ix_auth_sessions_user_id'), table_name='auth_sessions')
    op.drop_index(op.f('ix_auth_sessions_token_hash'), table_name='auth_sessions')
    op.drop_table('auth_sessions')
    op.drop_index(op.f('ix_auth_identities_user_id'), table_name='auth_identities')
    op.drop_index(op.f('ix_auth_identities_provider'), table_name='auth_identities')
    op.drop_table('auth_identities')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_providers_code'), table_name='providers')
    op.drop_table('providers')
