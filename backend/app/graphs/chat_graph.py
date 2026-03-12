from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict
from uuid import uuid4

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.constants import CHAT_THREAD_TITLE_MAX_LENGTH
from app.models import (
    ChatMessage,
    ChatThread,
    MessageRole,
    Provider,
    ProviderApiKey,
    ProviderCode,
    ProviderModel,
    utc_now,
)
from app.schemas import ChatRequest
from app.security import EncryptionConfigError, decrypt_secret

CONFIG_DB_KEY = "db"
ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Default to plain text answers, unless a specific format is requested or required. "
    "When user asks for markdown use four tildes as the opening/closing fence markers: ~~~~ ... ~~~~. "
    "Inside this, use normal Markdown/code syntax, including regular backticks where appropriate."
)


class ChatGraphState(TypedDict, total=False):
    prompt: str
    request_thread_id: str | None
    model_id: str | None
    provider_code: str | None
    user_id: str
    thread_id: str
    provider_id: int
    provider_api_key_id: str
    selected_provider_code: str
    selected_model_id: str
    error_message: str
    error_status: int


class ChatModelStreamState(TypedDict):
    prompt: str
    user_id: str
    thread_id: str
    provider_id: int
    provider_api_key_id: str
    selected_provider_code: str
    selected_model_id: str


@dataclass(frozen=True)
class ChatGraphResult:
    prompt: str
    user_id: str
    thread_id: str
    provider_id: int
    provider_api_key_id: str
    provider_code: ProviderCode
    model_id: str


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _build_thread_title(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    if not normalized:
        return "New chat"
    if len(normalized) <= CHAT_THREAD_TITLE_MAX_LENGTH:
        return normalized
    return f"{normalized[: CHAT_THREAD_TITLE_MAX_LENGTH - 1].rstrip()}…"


def _error_state(
    status_code: int,
    message: str,
) -> dict[str, str | int]:
    return {"error_message": message, "error_status": status_code}


def _route_after_step(state: ChatGraphState) -> Literal["continue", "error"]:
    # LangGraph merges partial node outputs into state, so successful nodes must keep
    # `error_status` reset to 0 to avoid stale error routing.
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


def _validate_request(state: ChatGraphState) -> dict[str, str | None | int]:
    prompt = (state.get("prompt") or "").strip()
    if not prompt:
        return _error_state(status.HTTP_422_UNPROCESSABLE_ENTITY, "Prompt cannot be blank.")

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
        "model_id": model_id,
        "provider_code": provider_code,
        "error_message": "",
        "error_status": 0,
    }


def _load_thread_history(
    state: ChatGraphState,
    config: RunnableConfig,
) -> dict[str, str]:
    db = _get_db(config)
    prompt = state["prompt"]
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
        thread = ChatThread(user_id=user_id, title=_build_thread_title(prompt))
        db.add(thread)
        db.commit()
        db.refresh(thread)

    return {"user_id": user_id, "thread_id": thread.id}


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


def _extract_chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)
    return ""


def _build_chat_client(
    provider_code: ProviderCode,
    model_name: str,
    api_key: str,
):
    if provider_code == ProviderCode.OPENAI:
        return ChatOpenAI(
            model=model_name,
            streaming=True,
            temperature=0.2,
            api_key=api_key,
        )

    if provider_code == ProviderCode.GROQ:
        return ChatOpenAI(
            model=model_name,
            streaming=True,
            temperature=0.2,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

    if provider_code == ProviderCode.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Anthropic chat provider is configured but `langchain-anthropic` is not installed. "
                    "Install it in the backend environment."
                ),
            ) from exc

        return ChatAnthropic(
            model=model_name,
            temperature=0.2,
            streaming=True,
            api_key=api_key,
        )

    if provider_code == ProviderCode.GEMINI:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Gemini chat provider is configured but `langchain-google-genai` is not installed. "
                    "Install it in the backend environment."
                ),
            ) from exc

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=api_key,
        )

    if provider_code == ProviderCode.XAI:
        return ChatOpenAI(
            model=model_name,
            streaming=True,
            temperature=0.2,
            base_url="https://api.x.ai/v1",
            api_key=api_key,
        )

    if provider_code == ProviderCode.OPENROUTER:
        return ChatOpenAI(
            model=model_name,
            streaming=True,
            temperature=0.2,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Provider `{provider_code.value}` is not supported for chat.",
    )


def _persist_assistant_message(
    db: Session,
    thread_id: str,
    content: str,
    model_name: str,
    provider_id: int,
) -> None:
    thread = db.get(ChatThread, thread_id)
    if thread is None:
        return

    db.add(
        ChatMessage(
            thread_id=thread_id,
            role=MessageRole.ASSISTANT,
            content=content,
            model_name=model_name,
            provider_id=provider_id,
        )
    )
    thread.updated_at = utc_now()
    db.commit()


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


def _persist_messages(
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

    if not thread.title:
        thread.title = _build_thread_title(state["prompt"])
    thread.provider_api_key_id = state["provider_api_key_id"]

    db.add(
        ChatMessage(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=state["prompt"],
            model_name=state["selected_model_id"],
            provider_id=state["provider_id"],
        )
    )
    thread.updated_at = utc_now()
    db.commit()
    return {}


def _error_node(_: ChatGraphState) -> dict[str, str]:
    return {}


async def _stream_model_and_persist(
    state: ChatModelStreamState,
    config: RunnableConfig,
) -> dict[str, str]:
    db = _get_db(config)
    writer = get_stream_writer()
    collected_parts: list[str] = []
    try:
        provider_code = ProviderCode(state["selected_provider_code"])
        api_key = _resolve_provider_api_key_value(
            db=db,
            user_id=state["user_id"],
            provider_id=state["provider_id"],
            provider_code=provider_code,
            provider_api_key_id=state["provider_api_key_id"],
        )
        llm = _build_chat_client(
            provider_code=provider_code,
            model_name=state["selected_model_id"],
            api_key=api_key,
        )

        async for chunk in llm.astream(
            [
                SystemMessage(content=ASSISTANT_SYSTEM_PROMPT),
                HumanMessage(content=state["prompt"]),
            ]
        ):
            chunk_text = _extract_chunk_text(chunk.content)
            if not chunk_text:
                continue
            collected_parts.append(chunk_text)
            writer(chunk_text)
    except Exception as exc:
        error_message = f"Error: {exc}"
        _persist_assistant_message(
            db=db,
            thread_id=state["thread_id"],
            content=error_message,
            model_name=state["selected_model_id"],
            provider_id=state["provider_id"],
        )
        writer(error_message)
    else:
        assistant_content = "".join(collected_parts).strip() or "(No response)"
        _persist_assistant_message(
            db=db,
            thread_id=state["thread_id"],
            content=assistant_content,
            model_name=state["selected_model_id"],
            provider_id=state["provider_id"],
        )
    return {}


def _build_chat_graph():
    graph = StateGraph(ChatGraphState)
    graph.add_node("validate_request", _validate_request)
    graph.add_node("load_thread_history", _load_thread_history)
    graph.add_node("resolve_user_provider_key", _resolve_user_provider_key)
    graph.add_node("persist_messages", _persist_messages)
    graph.add_node("error", _error_node)

    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges(
        "validate_request",
        _route_after_step,
        {
            "continue": "load_thread_history",
            "error": "error",
        },
    )
    graph.add_conditional_edges(
        "load_thread_history",
        _route_after_step,
        {
            "continue": "resolve_user_provider_key",
            "error": "error",
        },
    )
    graph.add_conditional_edges(
        "resolve_user_provider_key",
        _route_after_step,
        {
            "continue": "persist_messages",
            "error": "error",
        },
    )
    graph.add_edge("persist_messages", END)
    graph.add_edge("error", END)

    return graph.compile()


def _build_model_stream_graph():
    graph = StateGraph(ChatModelStreamState)
    graph.add_node("call_model", _stream_model_and_persist)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()


CHAT_GRAPH = _build_chat_graph()
CHAT_MODEL_STREAM_GRAPH = _build_model_stream_graph()


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


async def run_chat_graph(payload: ChatRequest, db: Session, *, user_id: str) -> ChatGraphResult:
    request_thread_id = _normalize_optional_text(payload.thread_id)
    config_thread_id = request_thread_id or f"new-thread-{uuid4()}"

    graph_input: ChatGraphState = {
        "prompt": payload.prompt,
        "request_thread_id": request_thread_id,
        "model_id": payload.model_id,
        "provider_code": payload.provider_code.value if payload.provider_code is not None else None,
        "user_id": user_id,
    }
    final_state = await CHAT_GRAPH.ainvoke(
        graph_input,
        config={
            "configurable": {
                "thread_id": config_thread_id,
                CONFIG_DB_KEY: db,
            }
        },
    )

    error_message = final_state.get("error_message")
    if error_message:
        error_status = final_state.get("error_status")
        if not isinstance(error_status, int) or error_status <= 0:
            error_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(status_code=error_status, detail=error_message)

    selected_provider_code = ProviderCode(_require_str(final_state, "selected_provider_code"))
    return ChatGraphResult(
        prompt=_require_str(final_state, "prompt"),
        user_id=_require_str(final_state, "user_id"),
        thread_id=_require_str(final_state, "thread_id"),
        provider_id=_require_int(final_state, "provider_id"),
        provider_api_key_id=_require_str(final_state, "provider_api_key_id"),
        provider_code=selected_provider_code,
        model_id=_require_str(final_state, "selected_model_id"),
    )


def build_model_stream_input(result: ChatGraphResult) -> ChatModelStreamState:
    return {
        "prompt": result.prompt,
        "user_id": result.user_id,
        "thread_id": result.thread_id,
        "provider_id": result.provider_id,
        "provider_api_key_id": result.provider_api_key_id,
        "selected_provider_code": result.provider_code.value,
        "selected_model_id": result.model_id,
    }
