from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import select, text
from sqlalchemy.orm import Session, joinedload

from app.constants import CHAT_THREAD_TITLE_MAX_LENGTH
from app.db import SessionLocal, init_db
from app.models import ChatMessage, ChatThread, MessageRole, Provider, ProviderApiKey, ProviderCode, ProviderModel, utc_now
from app.routers.catalog import router as catalog_router
from app.routers.users import router as users_router
from app.schemas import ChatRequest
from app.security import EncryptionConfigError, decrypt_secret
from app.services.current_user import get_dev_user_or_404

CHAT_MODEL = "gpt-5.2"
ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Default to plain text answers, unless a specific format is requested or required. "
    "When user asks for markdown use four tildes as the opening/closing fence markers: ~~~~ ... ~~~~. "
    "Inside this, use normal Markdown/code syntax, including regular backticks where appropriate."
)

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(catalog_router)


def _build_thread_title(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    if not normalized:
        return "New chat"
    if len(normalized) <= CHAT_THREAD_TITLE_MAX_LENGTH:
        return normalized
    return f"{normalized[: CHAT_THREAD_TITLE_MAX_LENGTH - 1].rstrip()}…"


def _extract_chunk_text(content: Any) -> str:
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


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_provider_model(
    db: Session,
    model_id: str | None,
    provider_code: ProviderCode | None,
) -> tuple[Provider | None, str]:
    if model_id is None:
        if provider_code is not None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="`provider_code` requires `model_id`.",
            )
        return None, CHAT_MODEL

    query = (
        select(ProviderModel)
        .join(Provider, Provider.id == ProviderModel.provider_id)
        .where(
            ProviderModel.model_id == model_id,
            ProviderModel.is_active.is_(True),
            Provider.is_active.is_(True),
        )
        .options(joinedload(ProviderModel.provider))
    )
    if provider_code is not None:
        query = query.where(Provider.code == provider_code)

    provider_model = db.scalar(query)
    if provider_model is None or provider_model.provider is None:
        detail = (
            f"Model `{model_id}` is not available for provider `{provider_code.value}`."
            if provider_code is not None
            else f"Model `{model_id}` is not available."
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

    return provider_model.provider, provider_model.model_id


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


def _resolve_api_key(
    db: Session,
    user_id: str,
    provider: Provider,
    thread: ChatThread,
) -> tuple[ProviderApiKey | None, str | None]:
    api_key_record = _find_user_provider_api_key(db, user_id, provider.id, thread)
    if api_key_record is not None:
        try:
            return api_key_record, decrypt_secret(api_key_record.encrypted_api_key)
        except (EncryptionConfigError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Stored provider API key could not be decrypted.",
            ) from exc

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            f"No active user API key found for provider `{provider.code.value}`. "
            "Add one in Settings to continue."
        ),
    )

def _build_chat_client(
    provider_code: ProviderCode | None,
    model_name: str,
    api_key: str | None,
) -> Any:
    selected_provider = provider_code or ProviderCode.OPENAI

    if selected_provider == ProviderCode.OPENAI:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "streaming": True,
            "temperature": 0.2,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    if selected_provider == ProviderCode.GROQ:
        kwargs = {
            "model": model_name,
            "streaming": True,
            "temperature": 0.2,
            "base_url": "https://api.groq.com/openai/v1",
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    if selected_provider == ProviderCode.ANTHROPIC:
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

        kwargs = {
            "model": model_name,
            "temperature": 0.2,
            "streaming": True,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if selected_provider == ProviderCode.GEMINI:
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

        kwargs = {
            "model": model_name,
            "temperature": 0.2,
        }
        if api_key:
            kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if selected_provider == ProviderCode.XAI:
        kwargs = {
            "model": model_name,
            "streaming": True,
            "temperature": 0.2,
            "base_url": "https://api.x.ai/v1",
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    if selected_provider == ProviderCode.OPENROUTER:
        kwargs = {
            "model": model_name,
            "streaming": True,
            "temperature": 0.2,
            "base_url": "https://openrouter.ai/api/v1",
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Provider `{selected_provider.value}` is not supported for chat.",
    )


def _get_or_create_thread(db: Session, user_id: str, prompt: str, thread_id: str | None) -> ChatThread:
    if thread_id:
        thread = db.scalar(
            select(ChatThread).where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
        )
        if thread is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat thread not found.")
        return thread

    thread = ChatThread(user_id=user_id, title=_build_thread_title(prompt))
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


@app.get("/health")
def health() -> dict[str, str]:
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/chat")
async def chat(payload: ChatRequest):
    normalized_prompt = payload.prompt.strip()
    thread_id = payload.thread_id.strip() if payload.thread_id else None
    model_id = _normalize_optional_text(payload.model_id)
    provider_code = payload.provider_code
    if not normalized_prompt:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Prompt cannot be blank.")

    db = SessionLocal()
    try:
        user = get_dev_user_or_404(db)
        thread = _get_or_create_thread(db, user.id, normalized_prompt, thread_id)
        provider, selected_model_name = _resolve_provider_model(db, model_id, provider_code)

        selected_provider_id: int | None = provider.id if provider is not None else None
        selected_provider_code: ProviderCode | None = provider.code if provider is not None else None
        selected_api_key: str | None = None
        selected_provider_api_key: ProviderApiKey | None = None
        if provider is not None:
            selected_provider_api_key, selected_api_key = _resolve_api_key(db, user.id, provider, thread)

        if not thread.title:
            thread.title = _build_thread_title(normalized_prompt)
        if provider is not None:
            thread.provider_api_key_id = selected_provider_api_key.id if selected_provider_api_key else None

        db.add(
            ChatMessage(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=normalized_prompt,
                model_name=selected_model_name,
                provider_id=selected_provider_id,
            )
        )
        thread.updated_at = utc_now()
        db.commit()
    except Exception:
        db.close()
        raise

    llm = _build_chat_client(selected_provider_code, selected_model_name, selected_api_key)

    async def stream_llm() -> AsyncIterator[bytes]:
        collected_parts: list[str] = []
        try:
            async for chunk in llm.astream(
                [
                    SystemMessage(content=ASSISTANT_SYSTEM_PROMPT),
                    HumanMessage(content=normalized_prompt),
                ]
            ):
                chunk_text = _extract_chunk_text(chunk.content)
                if not chunk_text:
                    continue
                collected_parts.append(chunk_text)
                yield chunk_text.encode("utf-8")
        except Exception as exc:
            error_message = f"Error: {exc}"
            db.add(
                ChatMessage(
                    thread_id=thread.id,
                    role=MessageRole.ASSISTANT,
                    content=error_message,
                    model_name=selected_model_name,
                    provider_id=selected_provider_id,
                )
            )
            thread.updated_at = utc_now()
            db.commit()
            yield error_message.encode("utf-8")
        else:
            assistant_content = "".join(collected_parts).strip() or "(No response)"
            db.add(
                ChatMessage(
                    thread_id=thread.id,
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    model_name=selected_model_name,
                    provider_id=selected_provider_id,
                )
            )
            thread.updated_at = utc_now()
            db.commit()
        finally:
            db.close()

    return StreamingResponse(
        stream_llm(),
        media_type="text/plain",
        headers={"X-Thread-Id": thread.id},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
