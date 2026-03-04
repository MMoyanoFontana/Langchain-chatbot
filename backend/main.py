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

from app.db import SessionLocal, init_db
from app.models import ChatMessage, ChatThread, MessageRole, utc_now
from app.routers.catalog import router as catalog_router
from app.routers.users import _get_dev_user_or_404, router as users_router

CHAT_MODEL = "gpt-4o-mini"
THREAD_TITLE_MAX_LENGTH = 120
ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked for or using Markdown, "
    "Always use four backticks instead of the usual three to enclose Markdown documents"
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

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, streaming=True)


def _build_thread_title(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    if not normalized:
        return "New chat"
    if len(normalized) <= THREAD_TITLE_MAX_LENGTH:
        return normalized
    return f"{normalized[: THREAD_TITLE_MAX_LENGTH - 1].rstrip()}…"


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


def _get_or_create_thread(db, user_id: str, prompt: str, thread_id: str | None) -> ChatThread:
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


@app.get("/chat")
async def chat(prompt: str, thread_id: str | None = None):
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Prompt cannot be blank.")

    db = SessionLocal()
    try:
        user = _get_dev_user_or_404(db)
        thread = _get_or_create_thread(db, user.id, normalized_prompt, thread_id)

        if not thread.title:
            thread.title = _build_thread_title(normalized_prompt)

        db.add(
            ChatMessage(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=normalized_prompt,
                model_name=CHAT_MODEL,
            )
        )
        thread.updated_at = utc_now()
        db.commit()
    except Exception:
        db.close()
        raise

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
                    model_name=CHAT_MODEL,
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
                    model_name=CHAT_MODEL,
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
