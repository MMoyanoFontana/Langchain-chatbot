from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProviderCode(str, enum.Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    XAI = "xai"
    OPENROUTER = "openrouter"
    OTHER = "other"


class MessageRole(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    provider_api_keys: Mapped[list["ProviderApiKey"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    chat_threads: Mapped[list["ChatThread"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[ProviderCode] = mapped_column(
        Enum(ProviderCode, name="provider_code"),
        unique=True,
        index=True,
    )
    display_name: Mapped[str] = mapped_column(String(80))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    api_keys: Mapped[list["ProviderApiKey"]] = relationship(back_populates="provider")
    models: Mapped[list["ProviderModel"]] = relationship(
        back_populates="provider",
        cascade="all, delete-orphan",
    )
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="provider")


class ProviderApiKey(Base):
    __tablename__ = "provider_api_keys"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "provider_id",
            "key_name",
            name="uq_provider_api_key_user_provider_name",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    provider_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )
    key_name: Mapped[str] = mapped_column(String(100), default="default")
    encrypted_api_key: Mapped[str] = mapped_column(Text)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    user: Mapped["User"] = relationship(back_populates="provider_api_keys")
    provider: Mapped["Provider"] = relationship(back_populates="api_keys")
    chat_threads: Mapped[list["ChatThread"]] = relationship(back_populates="provider_api_key")


class ProviderModel(Base):
    __tablename__ = "provider_models"
    __table_args__ = (
        UniqueConstraint(
            "provider_id",
            "model_id",
            name="uq_provider_model_provider_id_model_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(120))
    display_name: Mapped[str] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    provider: Mapped["Provider"] = relationship(back_populates="models")


class ChatThread(Base):
    __tablename__ = "chat_threads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    provider_api_key_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("provider_api_keys.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    user: Mapped["User"] = relationship(back_populates="chat_threads")
    provider_api_key: Mapped["ProviderApiKey | None"] = relationship(back_populates="chat_threads")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_threads.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"), index=True)
    content: Mapped[str] = mapped_column(Text)
    provider_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("providers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    model_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    thread: Mapped["ChatThread"] = relationship(back_populates="messages")
    provider: Mapped["Provider | None"] = relationship(back_populates="messages")

    @property
    def provider_code(self) -> ProviderCode | None:
        if self.provider is None:
            return None
        return self.provider.code
