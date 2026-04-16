from __future__ import annotations

import json
import uuid as _uuid
from dataclasses import dataclass
from typing import Literal, TypedDict

from fastapi import HTTPException, status
from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.constants import CHAT_THREAD_TITLE_MAX_LENGTH
from app.models import (
    ChatMessage,
    ChatThread,
    DocumentIndexStatus,
    IndexedDocument,
    MessageRole,
    Provider,
    ProviderApiKey,
    ProviderCode,
    ProviderModel,
    utc_now,
)
from app.schemas import ChatAttachment, ChatRequest
from app.security import EncryptionConfigError, decrypt_secret
from app.services.history import select_history_window
from app.services.memory import MemoryContext, get_memory_context
from app.services.rag import IngestionNotice, IngestionResult, RetrievalResult, get_rag_service

CONFIG_DB_KEY = "db"
CONFIG_RAG_SERVICE_KEY = "rag_service"
ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Default to plain text answers, unless a specific format is requested or required. "
    "When user asks for markdown use four tildes as the opening/closing fence markers: ~~~~ ... ~~~~. "
    "Inside this, use normal Markdown/code syntax, including regular backticks where appropriate. "
    "When uploaded-document context is provided, prefer it over unsupported inference and say so plainly if the files do not support a claim."
)


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------

class ChatGraphState(TypedDict, total=False):
    prompt: str
    attachments: list["ChatAttachmentState"]
    base_model_prompt: str
    model_prompt: str
    system_addendum: str
    thread_system_prompt: str | None
    request_system_prompt: str | None
    should_update_system_prompt: bool
    request_title: str | None
    history_messages: list["HistoryMessageState"]
    dropped_history_count: int
    dropped_history_message_ids: list[str]
    request_thread_id: str | None
    model_id: str | None
    provider_code: str | None
    user_id: str
    thread_id: str
    user_message_id: str
    provider_id: int
    provider_api_key_id: str
    selected_provider_code: str
    selected_model_id: str
    regenerate_from_message_id: str | None
    continue_from_message_id: str | None
    compare_with_user_message_id: str | None
    parent_message_id: str | None
    next_branch_index: int
    ingestion_notices: list["RagNoticeState"]
    pending_document_ids: list[str]
    retrieved_chunks: list["RetrievedChunkState"]
    retrieval_notice: str | None
    thread_summary: str | None
    user_memory_facts: list[dict[str, str]]
    thread_has_documents: bool
    error_message: str
    error_status: int


class ChatModelStreamState(TypedDict, total=False):
    prompt: str
    system_addendum: str
    thread_system_prompt: str | None
    history_messages: list["HistoryMessageState"]
    dropped_history_count: int
    dropped_history_message_ids: list[str]
    retrieved_chunks: list["RetrievedChunkState"]
    user_id: str
    thread_id: str
    provider_id: int
    provider_api_key_id: str
    selected_provider_code: str
    selected_model_id: str
    thread_has_documents: bool
    parent_message_id: str | None
    next_branch_index: int


@dataclass(frozen=True)
class ChatGraphResult:
    prompt: str
    system_addendum: str
    history_messages: tuple["HistoryMessageState", ...]
    dropped_history_count: int
    dropped_history_message_ids: tuple[str, ...]
    retrieved_chunks: tuple["RetrievedChunkState", ...]
    user_id: str
    thread_id: str
    provider_id: int
    provider_api_key_id: str
    provider_code: ProviderCode
    model_id: str
    thread_has_documents: bool
    pending_document_ids: tuple[str, ...] = ()
    parent_message_id: str | None = None
    next_branch_index: int = 0
    user_message_id: str | None = None
    thread_system_prompt: str | None = None


class ChatAttachmentState(TypedDict):
    type: Literal["file"]
    filename: str | None
    media_type: str
    url: str


class RagNoticeState(TypedDict):
    filename: str | None
    message: str


class RetrievedChunkState(TypedDict):
    document_id: str
    filename: str | None
    chunk_index: int
    score: float | None
    text: str


class HistoryMessageState(TypedDict):
    role: Literal["user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _error_state(status_code: int, message: str) -> dict[str, str | int]:
    return {"error_message": message, "error_status": status_code}


def _route_after_step(state: ChatGraphState) -> str:
    error_status = state.get("error_status")
    if isinstance(error_status, int) and error_status > 0:
        return "error"
    return "error" if state.get("error_message") else "continue"


def _get_db(config: RunnableConfig) -> Session:
    db = config.get("configurable", {}).get(CONFIG_DB_KEY)
    if not isinstance(db, Session):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Graph runtime database session is missing.",
        )
    return db


def _get_rag_service(config: RunnableConfig):
    rag_service = config.get("configurable", {}).get(CONFIG_RAG_SERVICE_KEY)
    if rag_service is not None:
        return rag_service
    return get_rag_service()


def _require_str(state: ChatGraphState, key: str) -> str:
    value = state.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph did not produce required field `{key}`.",
        )
    return value


def _require_int(state: ChatGraphState, key: str) -> int:
    value = state.get(key)
    if not isinstance(value, int):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph did not produce required field `{key}`.",
        )
    return value


# ---------------------------------------------------------------------------
# Prompt / attachment helpers
# ---------------------------------------------------------------------------

def _serialize_attachments(
    attachments: list[ChatAttachment] | None,
) -> list[ChatAttachmentState]:
    serialized: list[ChatAttachmentState] = []
    for attachment in attachments or []:
        media_type = attachment.media_type.strip()
        url = attachment.url.strip()
        if not media_type or not url:
            continue
        serialized.append(
            {
                "type": "file",
                "filename": _normalize_optional_text(attachment.filename),
                "media_type": media_type,
                "url": url,
            }
        )
    return serialized


def _format_attachment_label(attachment: ChatAttachmentState) -> str:
    filename = _normalize_optional_text(attachment.get("filename"))
    media_type = _normalize_optional_text(attachment.get("media_type"))
    if filename and media_type:
        return f"{filename} [{media_type}]"
    if filename:
        return filename
    if media_type:
        return media_type
    return "Attachment"


def _build_attachment_prompt(attachments: list[ChatAttachmentState]) -> str:
    if not attachments:
        return ""
    lines = ["Attached files:"]
    lines.extend(f"- {_format_attachment_label(a)}" for a in attachments)
    return "\n".join(lines)


def _build_model_prompt(prompt: str, attachments: list[ChatAttachmentState]) -> str:
    attachment_prompt = _build_attachment_prompt(attachments)
    parts = [part for part in [prompt, attachment_prompt] if part]
    return "\n\n".join(parts)


def _build_thread_title(prompt: str, attachments: list[ChatAttachmentState]) -> str:
    normalized = " ".join(prompt.strip().split())
    if not normalized and attachments:
        first_label = _format_attachment_label(attachments[0])
        normalized = (
            first_label
            if len(attachments) == 1
            else f"{first_label} + {len(attachments) - 1} more"
        )
    if not normalized:
        return "New chat"
    if len(normalized) <= CHAT_THREAD_TITLE_MAX_LENGTH:
        return normalized
    return f"{normalized[: CHAT_THREAD_TITLE_MAX_LENGTH - 1].rstrip()}…"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _find_user_provider_api_key(
    db: Session,
    user_id: str,
    provider_id: int,
    thread: ChatThread,
) -> ProviderApiKey | None:
    if thread.provider_api_key_id:
        existing_thread_key = db.scalar(
            select(ProviderApiKey).where(
                ProviderApiKey.id == thread.provider_api_key_id,
                ProviderApiKey.user_id == user_id,
                ProviderApiKey.provider_id == provider_id,
                ProviderApiKey.is_active.is_(True),
            )
        )
        if existing_thread_key is not None:
            return existing_thread_key

    return db.scalar(
        select(ProviderApiKey)
        .where(
            ProviderApiKey.user_id == user_id,
            ProviderApiKey.provider_id == provider_id,
            ProviderApiKey.is_active.is_(True),
        )
        .order_by(ProviderApiKey.is_default.desc(), ProviderApiKey.updated_at.desc())
    )


def _persist_assistant_message(
    db: Session,
    thread_id: str,
    content: str,
    model_name: str,
    provider_id: int,
    citations: list[RetrievedChunkState] | tuple[RetrievedChunkState, ...] | None = None,
    parent_message_id: str | None = None,
    branch_index: int = 0,
    metrics: dict | None = None,
) -> str | None:
    thread = db.get(ChatThread, thread_id)
    if thread is None:
        return None
    message_id = str(_uuid.uuid4())
    metrics = metrics or {}
    db.add(
        ChatMessage(
            id=message_id,
            thread_id=thread_id,
            role=MessageRole.ASSISTANT,
            content=content,
            citations=list(citations or []),
            model_name=model_name,
            provider_id=provider_id,
            parent_message_id=parent_message_id,
            branch_index=branch_index,
            prompt_tokens=metrics.get("prompt_tokens"),
            completion_tokens=metrics.get("completion_tokens"),
            total_tokens=metrics.get("total_tokens"),
            latency_ms=metrics.get("latency_ms"),
            time_to_first_token_ms=metrics.get("time_to_first_token_ms"),
        )
    )
    thread.updated_at = utc_now()
    db.commit()
    return message_id


def _persist_tool_message(
    db: Session,
    thread_id: str,
    tool_name: str,
    tool_input: dict,
    tool_output: str,
    model_name: str,
    provider_id: int,
) -> None:
    thread = db.get(ChatThread, thread_id)
    if thread is None:
        return
    db.add(
        ChatMessage(
            thread_id=thread_id,
            role=MessageRole.TOOL,
            content=json.dumps(
                {"tool": tool_name, "input": tool_input, "output": tool_output},
                ensure_ascii=False,
            ),
            model_name=model_name,
            provider_id=provider_id,
        )
    )
    thread.updated_at = utc_now()
    db.commit()


def _resolve_provider_api_key_value(
    db: Session,
    user_id: str,
    provider_id: int,
    provider_code: ProviderCode,
    provider_api_key_id: str,
) -> str:
    api_key_record = db.scalar(
        select(ProviderApiKey).where(
            ProviderApiKey.id == provider_api_key_id,
            ProviderApiKey.user_id == user_id,
            ProviderApiKey.provider_id == provider_id,
            ProviderApiKey.is_active.is_(True),
        )
    )
    if api_key_record is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No active user API key found for provider `{provider_code.value}`. "
                "Add one in Settings to continue."
            ),
        )
    try:
        return decrypt_secret(api_key_record.encrypted_api_key)
    except (EncryptionConfigError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored provider API key could not be decrypted.",
        ) from exc


# ---------------------------------------------------------------------------
# RAG serialization helpers
# ---------------------------------------------------------------------------

def _serialize_ingestion_notices(
    notices: tuple[IngestionNotice, ...],
) -> list[RagNoticeState]:
    return [
        {"filename": notice.filename, "message": notice.message}
        for notice in notices
    ]


def _deserialize_ingestion_result(notices: list[RagNoticeState]) -> IngestionResult:
    return IngestionResult(
        notices=tuple(
            IngestionNotice(
                filename=notice.get("filename"),
                message=notice["message"],
            )
            for notice in notices
        )
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _build_memory_addendum(
    thread_summary: str | None,
    user_facts: list[dict[str, str]],
) -> str:
    """Build a system-prompt addendum carrying memory context.

    Returns an empty string when there is nothing to add. The addendum is
    appended to the system prompt, NOT the user message, so memory entries
    cannot be misread as conversation content.
    """
    parts: list[str] = []
    if thread_summary:
        parts.append(f"[Previous conversation summary]\n{thread_summary}")
    if user_facts:
        fact_lines = [
            f"- {f['key']}: {f['value']}"
            for f in user_facts
            if f.get("key") and f.get("value")
        ]
        if fact_lines:
            parts.append("[About the user]\n" + "\n".join(fact_lines))
    return "\n\n".join(parts)


def _inject_memory_context(
    model_prompt: str,
    thread_summary: str | None,
    user_facts: list[dict[str, str]],
) -> str:
    """Backward-compatible helper used by older callers/tests.

    Prepends the memory addendum to the user prompt. New code should prefer
    `_build_memory_addendum` and place the result in the system prompt instead.
    """
    addendum = _build_memory_addendum(thread_summary, user_facts)
    if not addendum:
        return model_prompt
    return f"{addendum}\n\n{model_prompt}"


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------

def _validate_request(state: ChatGraphState) -> dict[str, str | None | int]:
    prompt = (state.get("prompt") or "").strip()
    attachments = state.get("attachments") or []
    is_regenerate = bool(state.get("regenerate_from_message_id"))
    is_compare = bool(state.get("compare_with_user_message_id"))
    base_model_prompt = _build_model_prompt(prompt, attachments)
    if not base_model_prompt and not is_regenerate and not is_compare:
        return _error_state(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Message text or at least one attachment is required.",
        )

    model_id = _normalize_optional_text(state.get("model_id"))
    if model_id is None:
        return _error_state(status.HTTP_422_UNPROCESSABLE_ENTITY, "Model is required.")

    provider_code = _normalize_optional_text(state.get("provider_code"))
    if provider_code is None:
        return _error_state(status.HTTP_422_UNPROCESSABLE_ENTITY, "Provider code is required.")
    if provider_code not in {item.value for item in ProviderCode}:
        return _error_state(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Provider `{provider_code}` is not supported.",
        )

    return {
        "prompt": prompt,
        "attachments": attachments,
        "base_model_prompt": base_model_prompt,
        "model_prompt": base_model_prompt,
        "model_id": model_id,
        "provider_code": provider_code,
        "ingestion_notices": [],
        "pending_document_ids": [],
        "retrieved_chunks": [],
        "retrieval_notice": None,
        "regenerate_from_message_id": state.get("regenerate_from_message_id"),
        "error_message": "",
        "error_status": 0,
    }


def _load_thread_history(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, str]:
    db = _get_db(config)
    prompt = state["prompt"]
    attachments = state.get("attachments") or []
    user_id = _normalize_optional_text(state.get("user_id"))
    if user_id is None:
        return _error_state(status.HTTP_401_UNAUTHORIZED, "Authentication is required.")
    request_thread_id = _normalize_optional_text(state.get("request_thread_id"))

    if request_thread_id:
        thread = db.scalar(
            select(ChatThread).where(
                ChatThread.id == request_thread_id,
                ChatThread.user_id == user_id,
            )
        )
        if thread is None:
            return _error_state(status.HTTP_404_NOT_FOUND, "Chat thread not found.")
    else:
        request_title = _normalize_optional_text(state.get("request_title"))
        title = request_title if request_title else _build_thread_title(prompt, attachments)
        thread = ChatThread(user_id=user_id, title=title)
        db.add(thread)
        db.commit()
        db.refresh(thread)

    if state.get("should_update_system_prompt"):
        raw = state.get("request_system_prompt")
        thread.system_prompt = raw.strip() if isinstance(raw, str) and raw.strip() else None
        db.commit()
        db.refresh(thread)

    return {"user_id": user_id, "thread_id": thread.id, "thread_system_prompt": thread.system_prompt}


def _resolve_user_provider_key(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, str | int]:
    db = _get_db(config)
    user_id = state["user_id"]
    thread_id = state["thread_id"]
    model_id = state["model_id"]
    provider_code_raw = state["provider_code"]

    if model_id is None or provider_code_raw is None:
        return _error_state(status.HTTP_422_UNPROCESSABLE_ENTITY, "Model and provider are required.")

    provider_code = ProviderCode(provider_code_raw)
    provider_model = db.scalar(
        select(ProviderModel)
        .join(Provider, Provider.id == ProviderModel.provider_id)
        .where(
            ProviderModel.model_id == model_id,
            ProviderModel.is_active.is_(True),
            Provider.code == provider_code,
            Provider.is_active.is_(True),
        )
        .options(joinedload(ProviderModel.provider))
    )
    if provider_model is None or provider_model.provider is None:
        return _error_state(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Model `{model_id}` is not available for provider `{provider_code.value}`.",
        )

    thread = db.scalar(
        select(ChatThread).where(
            ChatThread.id == thread_id,
            ChatThread.user_id == user_id,
        )
    )
    if thread is None:
        return _error_state(status.HTTP_404_NOT_FOUND, "Chat thread not found.")

    api_key = _find_user_provider_api_key(db, user_id, provider_model.provider_id, thread)
    if api_key is None:
        return _error_state(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            (
                f"No active user API key found for provider `{provider_model.provider.code.value}`. "
                "Add one in Settings to continue."
            ),
        )

    return {
        "provider_id": provider_model.provider.id,
        "provider_api_key_id": api_key.id,
        "selected_provider_code": provider_model.provider.code.value,
        "selected_model_id": provider_model.model_id,
    }


def _persist_user_message(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, str | int]:
    db = _get_db(config)

    thread = db.scalar(
        select(ChatThread).where(
            ChatThread.id == state["thread_id"],
            ChatThread.user_id == state["user_id"],
        )
    )
    if thread is None:
        return _error_state(status.HTTP_404_NOT_FOUND, "Chat thread not found.")

    thread.provider_api_key_id = state["provider_api_key_id"]

    # Compare mode: target an existing user message and create a sibling
    # assistant branch under it. The user message itself is not duplicated.
    compare_with = state.get("compare_with_user_message_id")
    if compare_with:
        target_user = db.get(ChatMessage, compare_with)
        if target_user is None or target_user.role != MessageRole.USER:
            return _error_state(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Cannot compare: target user message not found.",
            )
        max_branch = db.scalar(
            select(func.max(ChatMessage.branch_index)).where(
                ChatMessage.thread_id == state["thread_id"],
                ChatMessage.parent_message_id == target_user.id,
                ChatMessage.role == MessageRole.ASSISTANT,
            )
        )
        next_branch = (max_branch + 1) if max_branch is not None else 0
        thread.updated_at = utc_now()
        db.commit()
        return {
            "prompt": target_user.content,
            "base_model_prompt": _build_model_prompt(
                target_user.content,
                target_user.attachments or [],
            ),
            "model_prompt": _build_model_prompt(
                target_user.content,
                target_user.attachments or [],
            ),
            "user_message_id": target_user.id,
            "parent_message_id": target_user.id,
            "next_branch_index": next_branch,
            "error_message": "",
            "error_status": 0,
        }

    # When regenerating, reuse the existing last user message instead of
    # creating a duplicate row.
    if state.get("regenerate_from_message_id"):
        # Look up the specific assistant message being regenerated to find
        # its parent user message, rather than blindly using the last user
        # message in the thread.
        regen_assistant = db.get(ChatMessage, state["regenerate_from_message_id"])
        if regen_assistant is not None and regen_assistant.parent_message_id:
            last_user_message = db.get(ChatMessage, regen_assistant.parent_message_id)
        else:
            # Fallback for legacy messages without parent_message_id:
            # find the closest preceding user message.
            last_user_message = db.scalar(
                select(ChatMessage)
                .where(
                    ChatMessage.thread_id == state["thread_id"],
                    ChatMessage.role == MessageRole.USER,
                )
                .order_by(ChatMessage.created_at.desc())
                .limit(1)
            )
        if last_user_message is None:
            return _error_state(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Cannot regenerate: no prior user message found in this thread.",
            )

        # Determine next branch_index for the new sibling assistant message.
        max_branch = db.scalar(
            select(func.max(ChatMessage.branch_index)).where(
                ChatMessage.thread_id == state["thread_id"],
                ChatMessage.parent_message_id == last_user_message.id,
                ChatMessage.role == MessageRole.ASSISTANT,
            )
        )
        next_branch = (max_branch + 1) if max_branch is not None else 0

        thread.updated_at = utc_now()
        db.commit()
        # Use the existing message's content as the prompt for the model.
        return {
            "prompt": last_user_message.content,
            "base_model_prompt": _build_model_prompt(
                last_user_message.content,
                state.get("attachments") or [],
            ),
            "model_prompt": _build_model_prompt(
                last_user_message.content,
                state.get("attachments") or [],
            ),
            "user_message_id": last_user_message.id,
            "parent_message_id": last_user_message.id,
            "next_branch_index": next_branch,
            "error_message": "",
            "error_status": 0,
        }

    if not thread.title:
        thread.title = _build_thread_title(state["prompt"], state.get("attachments") or [])

    # Link user message to the assistant message it continues from, enabling
    # tree-structured conversation branching.
    continue_from = state.get("continue_from_message_id") or None

    message = ChatMessage(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=state["prompt"],
        attachments=state.get("attachments") or [],
        model_name=state["selected_model_id"],
        provider_id=state["provider_id"],
        parent_message_id=continue_from,
    )
    db.add(message)
    thread.updated_at = utc_now()
    db.commit()
    db.refresh(message)
    return {
        "user_message_id": message.id,
        "parent_message_id": message.id,
        "next_branch_index": 0,
        "error_message": "",
        "error_status": 0,
    }


async def _ingest_attachments(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, object]:
    attachments = state.get("attachments") or []
    if not attachments:
        return {
            "ingestion_notices": [],
            "pending_document_ids": [],
            "error_message": "",
            "error_status": 0,
        }

    rag_service = _get_rag_service(config)
    # Ingest synchronously so documents are INDEXED before _build_context_addendum
    # decides whether to bind the search_documents tool. Background indexing
    # races with the LLM stream and leaves the file unreachable on the same
    # turn it was uploaded.
    result = await rag_service.ingest_attachments(
        db=_get_db(config),
        user_id=state["user_id"],
        thread_id=state["thread_id"],
        message_id=state["user_message_id"],
        attachments=attachments,
        preferred_provider_api_key_id=state.get("provider_api_key_id"),
    )
    return {
        "ingestion_notices": _serialize_ingestion_notices(result.notices),
        "pending_document_ids": [],
        "error_message": "",
        "error_status": 0,
    }


async def _retrieve_context(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, object]:
    db = _get_db(config)
    rag_service = _get_rag_service(config)

    indexed_count = db.scalar(
        select(func.count(IndexedDocument.id)).where(
            IndexedDocument.user_id == state["user_id"],
            IndexedDocument.thread_id == state["thread_id"],
            IndexedDocument.status == DocumentIndexStatus.INDEXED,
        )
    ) or 0
    thread_has_documents = bool(indexed_count)

    # Memory context (summary + user facts) is placed in the system addendum
    # rather than prepended to the user prompt — keeps memory out of the
    # conversation surface so the LLM cannot mistake it for chat content.
    memory_addendum = _build_memory_addendum(
        state.get("thread_summary"),
        state.get("user_memory_facts") or [],
    )
    ingestion_result = _deserialize_ingestion_result(state.get("ingestion_notices") or [])
    ingestion_addendum = ""
    if ingestion_result.notices:
        lines = ["[Document indexing notices]"]
        for notice in ingestion_result.notices:
            label = notice.filename or "Attachment"
            lines.append(f"- {label}: {notice.message}")
        ingestion_addendum = "\n".join(lines)

    addendum_parts = [part for part in (memory_addendum, ingestion_addendum) if part]
    system_addendum = "\n\n".join(addendum_parts)

    return {
        "retrieved_chunks": [],
        "retrieval_notice": None,
        "thread_has_documents": thread_has_documents,
        "model_prompt": state["base_model_prompt"],
        "system_addendum": system_addendum,
        "error_message": "",
        "error_status": 0,
    }


def _load_history(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, object]:
    """Walk the branch lineage from the just-persisted user message and select
    a token-budgeted window of prior turns for the LLM."""
    db = _get_db(config)
    leaf_id = state.get("user_message_id")
    if not leaf_id:
        return {
            "history_messages": [],
            "dropped_history_count": 0,
            "dropped_history_message_ids": [],
        }

    window = select_history_window(db, leaf_message_id=leaf_id)
    return {
        "history_messages": [
            {"role": m.role, "content": m.content} for m in window.messages
        ],
        "dropped_history_count": window.dropped_count,
        "dropped_history_message_ids": list(window.dropped_message_ids),
    }


def _load_memory(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, object]:
    try:
        db = _get_db(config)
        ctx: MemoryContext = get_memory_context(
            db=db,
            thread_id=state["thread_id"],
            user_id=state["user_id"],
        )
        return {
            "thread_summary": ctx.summary,
            "user_memory_facts": ctx.user_facts,
        }
    except Exception:
        return {"thread_summary": None, "user_memory_facts": []}


def _error_node(_: ChatGraphState) -> dict[str, str]:
    return {}
