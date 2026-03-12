from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from sqlalchemy import text

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
from app.schemas import ChatRequest
from app.services.auth import purge_expired_sessions
from app.services.current_user import require_current_user

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
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Session-Token"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(catalog_router)


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
        graph_result = await run_chat_graph(payload=payload, db=db, user_id=user.id)
    except Exception:
        close_db_session()
        raise

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

    return StreamingResponse(
        stream_llm(),
        media_type="text/plain",
        headers={"X-Thread-Id": graph_result.thread_id},
        background=BackgroundTask(close_db_session),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
