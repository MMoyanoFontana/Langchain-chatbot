from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

from app.db import SessionLocal, init_db
from app.graphs.chat_graph import (
    CHAT_MODEL_STREAM_GRAPH,
    CONFIG_DB_KEY,
    build_model_stream_input,
    run_chat_graph,
)
from app.models import User
from app.routers.auth import router as auth_router
from app.routers.catalog import router as catalog_router
from app.routers.users import router as users_router
from app.runtime_config import get_cors_allowed_origins
from app.schemas import ChatRequest
from app.services.auth import purge_expired_sessions
from app.services.current_user import require_current_user
from app.services.rag import get_rag_service

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    with SessionLocal() as db:
        purge_expired_sessions(db)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Session-Token"],
    expose_headers=["X-Thread-Id", "X-User-Message-Id"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(catalog_router)


async def _index_documents_in_background(
    user_id: str, thread_id: str, document_ids: list[str]
) -> None:
    if not document_ids:
        return
    rag_service = get_rag_service()
    with SessionLocal() as db:
        for doc_id in document_ids:
            try:
                await rag_service.retry_document(
                    db=db, user_id=user_id, thread_id=thread_id, document_id=doc_id
                )
            except Exception:
                LOGGER.exception(
                    "background_index_failed user_id=%s thread_id=%s document_id=%s",
                    user_id, thread_id, doc_id,
                )


@app.get("/health")
def health() -> dict[str, str]:
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/chat")
async def chat(payload: ChatRequest, user: User = Depends(require_current_user)):
    db = SessionLocal()
    closed = False

    def close_db_session() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        db.close()

    try:
        graph_result = await run_chat_graph(
            payload=payload,
            db=db,
            user_id=user.id,
            rag_service=get_rag_service(),
        )
    except Exception:
        close_db_session()
        raise

    if graph_result.pending_document_ids:
        asyncio.ensure_future(
            _index_documents_in_background(
                user_id=graph_result.user_id,
                thread_id=graph_result.thread_id,
                document_ids=list(graph_result.pending_document_ids),
            )
        )

    stream_input = build_model_stream_input(graph_result)
    stream_config = {
        "configurable": {
            "thread_id": graph_result.thread_id,
            CONFIG_DB_KEY: db,
        }
    }

    async def stream_llm() -> AsyncIterator[bytes]:
        try:
            async for chunk in CHAT_MODEL_STREAM_GRAPH.astream(
                stream_input,
                config=stream_config,
                stream_mode="custom",
            ):
                text_chunk = chunk if isinstance(chunk, str) else str(chunk)
                if not text_chunk:
                    continue
                yield text_chunk.encode("utf-8")
        finally:
            close_db_session()

    response_headers = {"X-Thread-Id": graph_result.thread_id}
    if graph_result.user_message_id:
        response_headers["X-User-Message-Id"] = graph_result.user_message_id

    return StreamingResponse(
        stream_llm(),
        media_type="text/plain",
        headers=response_headers,
        background=BackgroundTask(close_db_session),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
