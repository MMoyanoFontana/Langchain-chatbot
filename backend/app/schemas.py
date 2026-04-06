from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.constants import CHAT_THREAD_TITLE_MAX_LENGTH
from app.models import AuthProvider, DocumentIndexStatus, MessageRole, ProviderCode


class UserUpdate(BaseModel):
    email: str | None = Field(default=None, min_length=3, max_length=320)
    full_name: str | None = Field(default=None, max_length=120)
    is_active: bool | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str | None
    avatar_url: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class EmailRegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=120)


class EmailLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class AuthProviderRead(BaseModel):
    code: AuthProvider
    label: str
    enabled: bool


class OAuthAuthorizeRead(BaseModel):
    authorize_url: str


class OAuthExchangeRequest(BaseModel):
    code: str = Field(min_length=1, max_length=4096)
    state: str = Field(min_length=1, max_length=4096)
    redirect_uri: str = Field(min_length=1, max_length=2048)


class AuthSessionRead(BaseModel):
    session_token: str
    user: UserRead
    redirect_to: str | None = None


class ProviderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: ProviderCode
    display_name: str
    is_active: bool


class ProviderSettingsRead(BaseModel):
    provider: ProviderRead
    has_key: bool
    default_key_name: str | None = None


class ProviderApiKeyUpsert(BaseModel):
    api_key: str = Field(min_length=1, max_length=4096)
    key_name: str = Field(default="default", min_length=1, max_length=100)
    is_default: bool = True
    is_active: bool = True


class ProviderApiKeyUpdate(BaseModel):
    api_key: str | None = Field(default=None, min_length=1, max_length=4096)
    key_name: str | None = Field(default=None, min_length=1, max_length=100)
    is_default: bool | None = None
    is_active: bool | None = None


class ProviderApiKeyRead(BaseModel):
    id: str
    user_id: str
    key_name: str
    is_default: bool
    is_active: bool
    masked_api_key: str
    provider: ProviderRead
    created_at: datetime
    updated_at: datetime


class ProviderModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: str
    display_name: str
    is_active: bool
    provider: ProviderRead


class ChatAttachment(BaseModel):
    type: Literal["file"] = "file"
    filename: str | None = Field(default=None, max_length=512)
    media_type: str = Field(
        min_length=1,
        validation_alias=AliasChoices("media_type", "mediaType"),
    )
    url: str = Field(min_length=1)


class ChatRequest(BaseModel):
    prompt: str = Field(default="")
    thread_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
    )
    model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model_id", "modelId"),
    )
    provider_code: ProviderCode | None = Field(
        default=None,
        validation_alias=AliasChoices("provider_code", "providerCode"),
    )
    attachments: list[ChatAttachment] = Field(default_factory=list)
    regenerate_from_message_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("regenerate_from_message_id", "regenerateFromMessageId"),
    )


class ChatCitation(BaseModel):
    document_id: str = Field(min_length=1, max_length=36)
    filename: str | None = Field(default=None, max_length=512)
    chunk_index: int = Field(ge=0)
    score: float | None = None
    text: str = Field(min_length=1)


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: MessageRole
    content: str
    attachments: list[ChatAttachment] = Field(default_factory=list)
    citations: list[ChatCitation] = Field(default_factory=list)
    provider_code: ProviderCode | None = None
    model_name: str | None
    branch_index: int = 0
    created_at: datetime


class ChatThreadDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_message_id: str | None
    filename: str | None
    media_type: str
    byte_size: int
    chunk_count: int
    status: DocumentIndexStatus
    error_message: str | None
    indexed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ChatThreadSummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class ChatThreadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    documents: list[ChatThreadDocumentRead] = Field(default_factory=list)
    messages: list[ChatMessageRead]


class ChatThreadUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=CHAT_THREAD_TITLE_MAX_LENGTH)


class UserMemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    value: str
    updated_at: datetime


class UserMemoryUpdate(BaseModel):
    value: str = Field(min_length=1, max_length=200)
