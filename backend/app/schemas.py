from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.constants import CHAT_THREAD_TITLE_MAX_LENGTH
from app.models import MessageRole, ProviderCode


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    full_name: str | None = Field(default=None, max_length=120)


class UserUpdate(BaseModel):
    email: str | None = Field(default=None, min_length=3, max_length=320)
    full_name: str | None = Field(default=None, max_length=120)
    is_active: bool | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


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


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
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


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: MessageRole
    content: str
    provider_code: ProviderCode | None = None
    model_name: str | None
    created_at: datetime


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
    messages: list[ChatMessageRead]


class ChatThreadUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=CHAT_THREAD_TITLE_MAX_LENGTH)
