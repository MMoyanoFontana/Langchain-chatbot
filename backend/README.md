# Backend

FastAPI + LangGraph + SQLAlchemy + Pinecone. See the [root README](../README.md) for architecture and product overview.

## Run

```bash
uv sync
python main.py                  # :8000, applies migrations on startup
```

## Test

```bash
uv run pytest tests/            # all tests
uv run pytest tests/test_rag_integration.py    # one file
uv run pytest -k tool_calling   # by name
```

Tests use in-memory SQLite (`StaticPool`) with a fake vector store and fake embeddings — no Pinecone or OpenAI calls. Each test gets a fresh schema.

## Database

**Local dev:** Defaults to SQLite (`sqlite:///./chatbot.db`), which requires no setup.

**Production:** Use Postgres via `DATABASE_URL`. Free-tier Neon setup:

1. Sign up at [neon.tech](https://neon.tech)
2. Create a new project
3. Copy your connection string (`postgres://user:pass@host/dbname`)
4. Set `DATABASE_URL` in your deploy environment (Render, Railway, etc.)

The app handles both SQLite and Postgres automatically—migrations run on startup, same code path.

### Migrations

Alembic migrations live in [alembic/versions/](alembic/versions/) and run automatically on startup via `init_db()`. Don't run `alembic upgrade head` manually unless you want to test migration logic in isolation.

After changing `app/models.py`:

```bash
alembic revision --autogenerate -m "describe the change"
# Review the generated file, rename if needed (YYYYMMDD_NNNNNN_description.py), commit
```

## Deployment (Render)

The `Dockerfile` and `render.yaml` at the repo root define the deployment:

1. **Set up Postgres:** Create a free Neon project, copy the connection string
2. **Create Render service:** Connect this repo to Render (via GitHub)
3. **Configure env vars** in Render dashboard:
   - `DATABASE_URL` → Neon connection string
   - `API_KEY_ENCRYPTION_KEY` → Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
   - `AUTH_SECRET` → Any strong random string
   - `BACKEND_CORS_ORIGINS` → Your frontend URL (e.g., `https://chatbot.vercel.app`)
   - Other optional vars: `PINECONE_*`, `LANGSMITH_*`, `GOOGLE_OAUTH_*`
4. **Deploy:** Render auto-deploys on push to main (or manually via dashboard)

Migrations run automatically on startup, so the schema is always current.

## Environment variables

Reads `.env` in this directory (loaded by `python-dotenv`).

### Required

| Variable | Notes |
|---|---|
| `API_KEY_ENCRYPTION_KEY` | Fernet key — `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `AUTH_SECRET` | Secret for session tokens and OAuth state |
| `BACKEND_CORS_ORIGINS` | Comma-separated origins, e.g. `http://localhost:3000` |

### Required for RAG

| Variable | Notes |
|---|---|
| `PINECONE_API_KEY` | |
| `PINECONE_INDEX_NAME` | Must exist on your Pinecone project before startup |
| `PINECONE_NAMESPACE_PREFIX` | Defaults to `rag` |
| `OPENAI_API_KEY` | Server-side embedding key. Falls back to the user's stored key if absent. |

### Optional

| Variable | Notes |
|---|---|
| `DATABASE_URL` | Defaults to `sqlite:///./chatbot.db`. Use a Postgres URL in prod. |
| `RATE_LIMIT_PER_MINUTE` | Per-user limit on `/chat` and document retry. Defaults to 60. |
| `CATALOG_SYNC_INTERVAL_HOURS` | Defaults to 24. |
| `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT` | Enables LangSmith tracing. |
| `GOOGLE_OAUTH_CLIENT_ID` / `_SECRET` | OAuth provider creds — repeat for `GITHUB_*` and `MICROSOFT_*`. |

## Layout

```
app/
  graphs/
    chat_graph.py        Prep graph: validate → load history → memory → persist → ingest → retrieve
    _stream.py           Streaming graph: model invocation + multi-turn tool loop
    _nodes.py            Shared node implementations
  routers/
    auth.py              Login, register, OAuth, sessions
    users.py             Threads, messages, documents, memories, provider keys
    catalog.py           Provider/model listing
  services/
    rag.py               Pinecone ingestion, retrieval, document delete/retry
    memory.py            Rolling thread summary + user-fact extraction
    auth.py              Password hashing, session tokens, OAuth flows
    catalog_sync.py      Periodic upstream model-list refresh
    pinecone_store.py    Vector store wrapper
  models.py              SQLAlchemy ORM
  schemas.py             Pydantic request/response
  security.py            Fernet encryption
alembic/versions/        Migrations
tests/                   pytest suite
main.py                  FastAPI app + /chat streaming endpoint
```

## Conventions

**Routers are thin.** Business logic lives in plain functions; route decorators wrap them. Lets us call the same logic from tests without spinning up `TestClient`.

```python
def delete_user_thread_document(user_id, thread_id, document_id, db):
    # logic here

@router.delete("/me/threads/{thread_id}/documents/{document_id}", status_code=204)
def delete_current_user_thread_document(thread_id, document_id, user=Depends(...), db=Depends(...)):
    return delete_user_thread_document(user.id, thread_id, document_id, db)
```

**Async RAG calls from sync routes.** Use `asyncio.run()` inside sync FastAPI handlers to call async RAG service methods. Async graph nodes use `await` directly.

**Graph error handling.** Nodes return `{"error_message": ..., "error_status": ...}` to signal errors; `_route_after_step` routes to the `error` node, which raises an `HTTPException`.
