from __future__ import annotations

import base64
from dataclasses import dataclass
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.graphs import chat_graph as chat_graph_module
from app.graphs._nodes import _persist_assistant_message as _nodes_persist_assistant_message
from app.graphs.chat_graph import run_chat_graph
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
    User,
    utc_now,
)
from app.routers import users as users_router
from app.schemas import ChatAttachment, ChatRequest
from app.services.pinecone_store import PineconeQueryMatch
from app.services.rag import IngestionResult, RagService, RagSettings, RetrievedChunk, RetrievalResult


def _session_factory():
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def _data_url(media_type: str, text: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(text.encode('utf-8')).decode('utf-8')}"


def _create_user(db: Session, *, user_id: str = "user-1") -> User:
    user = User(
        id=user_id,
        email=f"{user_id}@example.com",
        full_name="RAG User",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_openai_provider_stack(db: Session, *, user_id: str) -> None:
    provider = Provider(code=ProviderCode.OPENAI, display_name="OpenAI", is_active=True)
    db.add(provider)
    db.flush()

    db.add(
        ProviderModel(
            provider_id=provider.id,
            model_id="gpt-5-mini",
            display_name="GPT-5 mini",
            is_active=True,
        )
    )
    db.add(
        ProviderApiKey(
            user_id=user_id,
            provider_id=provider.id,
            key_name="default",
            encrypted_api_key="unused-for-run-chat-graph",
            is_default=True,
            is_active=True,
        )
    )
    db.commit()


class FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.5] for index, _ in enumerate(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


async def _fake_build_embeddings_client(**_: object) -> FakeEmbeddings:
    return FakeEmbeddings()


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[object]]] = []
        self.queries: list[tuple[str, list[float], int, dict[str, object] | None]] = []
        self.deletes: list[tuple[str, dict[str, object]]] = []
        self.query_results: list[PineconeQueryMatch] = []
        self.dimension = 1024

    async def upsert(self, *, namespace: str, vectors: list[object]) -> None:
        self.upserts.append((namespace, vectors))

    async def query(
        self,
        *,
        namespace: str,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[PineconeQueryMatch]:
        self.queries.append((namespace, query_vector, top_k, metadata_filter))
        return list(self.query_results)

    async def delete_by_filter(
        self,
        *,
        namespace: str,
        metadata_filter: dict[str, object],
    ) -> None:
        self.deletes.append((namespace, metadata_filter))

    async def describe_index_dimension(self) -> int:
        return self.dimension


@dataclass
class FakeGraphRagService:
    ingest_calls: list[tuple[str, str, str, int]]
    retrieve_calls: list[tuple[str, str, str]]

    async def ingest_attachments(
        self,
        *,
        db: Session,
        user_id: str,
        thread_id: str,
        message_id: str,
        attachments: list[dict[str, object]],
        preferred_provider_api_key_id: str | None,
    ) -> IngestionResult:
        self.ingest_calls.append((user_id, thread_id, message_id, len(attachments)))
        return IngestionResult()

    async def retrieve(
        self,
        *,
        db: Session,
        user_id: str,
        thread_id: str,
        prompt: str,
        preferred_provider_api_key_id: str | None,
    ) -> RetrievalResult:
        self.retrieve_calls.append((user_id, thread_id, prompt))
        return RetrievalResult(
            thread_has_documents=True,
            chunks=(
                RetrievedChunk(
                    document_id="doc-1",
                    filename="notes.txt",
                    chunk_index=0,
                    score=0.92,
                    text="The launch date is April 3.",
                ),
            ),
        )

    def build_augmented_prompt(
        self,
        *,
        base_prompt: str,
        ingestion_result: IngestionResult,
        retrieval_result: RetrievalResult,
    ) -> str:
        if retrieval_result.chunks:
            return f"{base_prompt}\n\nRetrieved uploaded-document context:\n{retrieval_result.chunks[0].text}"
        return base_prompt


def test_rag_service_ingests_attachment_and_deduplicates() -> None:
    SessionLocal = _session_factory()
    vector_store = FakeVectorStore()
    rag_service = RagService(
        settings=RagSettings(
            enabled=True,
            pinecone_api_key="test",
            pinecone_index_name="test-index",
            namespace_prefix="rag",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=None,
            chunk_size=80,
            chunk_overlap=10,
            top_k=4,
            max_context_chars=500,
            min_relevance_score=0.2,
            max_file_bytes=1024 * 1024,
        ),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    rag_service._build_embeddings_client = _fake_build_embeddings_client  # type: ignore[method-assign]

    with SessionLocal() as db:
        user = _create_user(db)
        thread = ChatThread(user_id=user.id, title="Docs")
        db.add(thread)
        db.flush()
        message = ChatMessage(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="Index this",
            attachments=[],
            model_name="gpt-5-mini",
            provider_id=None,
        )
        db.add(message)
        db.commit()
        db.refresh(thread)
        db.refresh(message)

        attachment = {
            "type": "file",
            "filename": "notes.txt",
            "media_type": "text/plain",
            "url": _data_url(
                "text/plain",
                "Alpha section.\n\nBeta section.\n\nGamma section with enough text to force chunking.",
            ),
        }

        import asyncio

        first_result = asyncio.run(
            rag_service.ingest_attachments(
                db=db,
                user_id=user.id,
                thread_id=thread.id,
                message_id=message.id,
                attachments=[attachment],
                preferred_provider_api_key_id=None,
            )
        )
        second_result = asyncio.run(
            rag_service.ingest_attachments(
                db=db,
                user_id=user.id,
                thread_id=thread.id,
                message_id=message.id,
                attachments=[attachment],
                preferred_provider_api_key_id=None,
            )
        )

        documents = list(db.scalars(select(IndexedDocument)).all())

        assert first_result.notices == ()
        assert second_result.notices == ()
        assert len(documents) == 1
        assert documents[0].status == DocumentIndexStatus.INDEXED
        assert documents[0].chunk_count >= 1
        assert len(vector_store.upserts) == 1


def test_rag_service_retrieval_scopes_query_to_thread() -> None:
    SessionLocal = _session_factory()
    vector_store = FakeVectorStore()
    vector_store.query_results = [
        PineconeQueryMatch(
            id="doc-1:0",
            score=0.95,
            metadata={
                "document_id": "doc-1",
                "thread_id": "thread-1",
                "filename": "scope.txt",
                "chunk_index": 0,
                "text": "Scoped answer.",
            },
        )
    ]
    rag_service = RagService(
        settings=RagSettings(
            enabled=True,
            pinecone_api_key="test",
            pinecone_index_name="test-index",
            namespace_prefix="rag",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=None,
            chunk_size=1200,
            chunk_overlap=200,
            top_k=4,
            max_context_chars=500,
            min_relevance_score=0.2,
            max_file_bytes=1024 * 1024,
        ),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    rag_service._build_embeddings_client = _fake_build_embeddings_client  # type: ignore[method-assign]

    with SessionLocal() as db:
        user = _create_user(db)
        thread = ChatThread(id="thread-1", user_id=user.id, title="Scoped thread")
        db.add(thread)
        db.flush()
        db.add(
            IndexedDocument(
                id="doc-1",
                user_id=user.id,
                thread_id=thread.id,
                source_message_id=None,
                filename="scope.txt",
                media_type="text/plain",
                checksum_sha256="abc123",
                byte_size=12,
                chunk_count=1,
                pinecone_namespace=f"rag-{user.id}",
                status=DocumentIndexStatus.INDEXED,
                error_message=None,
                indexed_at=utc_now(),
            )
        )
        db.commit()

        import asyncio

        result = asyncio.run(
            rag_service.retrieve(
                db=db,
                user_id=user.id,
                thread_id=thread.id,
                prompt="What does the file say?",
                preferred_provider_api_key_id=None,
            )
        )

        assert result.chunks
        assert vector_store.queries == [
            (
                f"rag-{user.id}",
                [0.1, 0.2],
                4,
                {"thread_id": {"$eq": thread.id}},
            )
        ]


def test_rag_service_matches_embedding_dimensions_to_pinecone(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class CapturingEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    vector_store = FakeVectorStore()
    vector_store.dimension = 1024
    rag_service = RagService(
        settings=RagSettings(
            enabled=True,
            pinecone_api_key="test",
            pinecone_index_name="test-index",
            namespace_prefix="rag",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=None,
            chunk_size=1200,
            chunk_overlap=200,
            top_k=4,
            max_context_chars=500,
            min_relevance_score=0.2,
            max_file_bytes=1024 * 1024,
        ),
        vector_store=vector_store,  # type: ignore[arg-type]
    )

    monkeypatch.setattr("app.services.rag.OpenAIEmbeddings", CapturingEmbeddings)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import asyncio

    asyncio.run(
        rag_service._build_embeddings_client(
            db=Session(bind=create_engine("sqlite://", future=True)),
            user_id="user-1",
            preferred_provider_api_key_id=None,
        )
    )

    assert captured["model"] == "text-embedding-3-small"
    assert captured["dimensions"] == 1024
    assert captured["api_key"] == "test-key"


def test_run_chat_graph_defers_retrieval_to_tool_call() -> None:
    """Prepare-graph phase no longer auto-retrieves; retrieval is deferred to the
    streaming phase via the search_documents tool.  The graph result must carry
    thread_has_documents=True so the streaming node can bind the tool, and
    retrieved_chunks must be empty (populated later when the tool runs).
    """
    import asyncio

    SessionLocal = _session_factory()

    with SessionLocal() as db:
        user = _create_user(db)
        _create_openai_provider_stack(db, user_id=user.id)
        rag_service = FakeGraphRagService(ingest_calls=[], retrieve_calls=[])

        graph_result = asyncio.run(
            run_chat_graph(
                ChatRequest(
                    prompt="When is the launch date?",
                    model_id="gpt-5-mini",
                    provider_code=ProviderCode.OPENAI,
                    attachments=[
                        ChatAttachment(
                            filename="notes.txt",
                            media_type="text/plain",
                            url=_data_url("text/plain", "Launch date: April 3."),
                        )
                    ],
                ),
                db,
                user_id=user.id,
                rag_service=rag_service,
            )
        )

        persisted_messages = list(db.scalars(select(ChatMessage).where(ChatMessage.thread_id == graph_result.thread_id)).all())

        # Ingestion still happens in prepare phase.
        assert len(rag_service.ingest_calls) == 1
        # Retrieval is NOT called during the prepare graph — deferred to streaming.
        assert len(rag_service.retrieve_calls) == 0
        # retrieved_chunks is empty in the graph result (tool call populates it later).
        assert graph_result.retrieved_chunks == ()
        # User message is persisted.
        assert len(persisted_messages) == 1
        assert persisted_messages[0].attachments[0]["filename"] == "notes.txt"


def test_thread_reads_include_assistant_citations() -> None:
    SessionLocal = _session_factory()

    with SessionLocal() as db:
        user = _create_user(db)
        _create_openai_provider_stack(db, user_id=user.id)
        provider = db.scalar(select(Provider).where(Provider.code == ProviderCode.OPENAI))
        assert provider is not None

        thread = ChatThread(user_id=user.id, title="Grounded answer")
        db.add(thread)
        db.commit()
        db.refresh(thread)

        _nodes_persist_assistant_message(
            db=db,
            thread_id=thread.id,
            content="The launch date is April 3.",
            model_name="gpt-5-mini",
            provider_id=provider.id,
            citations=[
                {
                    "document_id": "doc-1",
                    "filename": "notes.txt",
                    "chunk_index": 0,
                    "score": 0.92,
                    "text": "The launch date is April 3.",
                }
            ],
        )

        thread_read = users_router.get_user_thread(user.id, thread.id, db)

        assert len(thread_read.messages) == 1
        assert thread_read.messages[0].citations[0].document_id == "doc-1"
        assert thread_read.messages[0].citations[0].filename == "notes.txt"
        assert thread_read.messages[0].citations[0].chunk_index == 0
        assert thread_read.messages[0].citations[0].text == "The launch date is April 3."


def test_thread_reads_include_indexed_documents() -> None:
    SessionLocal = _session_factory()

    with SessionLocal() as db:
        user = _create_user(db)
        thread = ChatThread(user_id=user.id, title="Docs")
        db.add(thread)
        db.flush()
        db.add(
            IndexedDocument(
                id="doc-1",
                user_id=user.id,
                thread_id=thread.id,
                source_message_id=None,
                filename="notes.txt",
                media_type="text/plain",
                checksum_sha256="checksum-1",
                byte_size=256,
                chunk_count=3,
                pinecone_namespace=f"rag-{user.id}",
                status=DocumentIndexStatus.INDEXED,
                error_message=None,
                indexed_at=utc_now(),
            )
        )
        db.add(
            IndexedDocument(
                id="doc-2",
                user_id=user.id,
                thread_id=thread.id,
                source_message_id=None,
                filename="broken.pdf",
                media_type="application/pdf",
                checksum_sha256="checksum-2",
                byte_size=512,
                chunk_count=0,
                pinecone_namespace=f"rag-{user.id}",
                status=DocumentIndexStatus.FAILED,
                error_message="PDF text extraction returned no text.",
                indexed_at=None,
            )
        )
        db.commit()
        db.refresh(thread)

        thread_read = users_router.get_user_thread(user.id, thread.id, db)

        assert [document.id for document in thread_read.documents] == ["doc-2", "doc-1"]
        assert thread_read.documents[0].status == DocumentIndexStatus.FAILED
        assert thread_read.documents[0].error_message == "PDF text extraction returned no text."
        assert thread_read.documents[1].status == DocumentIndexStatus.INDEXED
        assert thread_read.documents[1].chunk_count == 3


def test_delete_user_thread_triggers_rag_cleanup(monkeypatch) -> None:
    SessionLocal = _session_factory()

    class CleanupRagService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def delete_thread_documents(self, *, db: Session, user_id: str, thread_id: str) -> None:
            self.calls.append((user_id, thread_id))

    cleanup_service = CleanupRagService()
    monkeypatch.setattr(users_router, "get_rag_service", lambda: cleanup_service)

    with SessionLocal() as db:
        user = _create_user(db)
        thread = ChatThread(user_id=user.id, title="Delete me")
        db.add(thread)
        db.commit()
        db.refresh(thread)

        response = users_router.delete_user_thread(user.id, thread.id, db)
        deleted_thread = db.scalar(select(ChatThread).where(ChatThread.id == thread.id))

        assert response.status_code == 204
        assert cleanup_service.calls == [(user.id, thread.id)]
        assert deleted_thread is None


def test_attachment_only_message_completes_prepare_graph() -> None:
    """An attachment-only message (no text prompt) must go through the prepare graph
    successfully. The LLM will decide whether to call search_documents at streaming
    time — no retrieval happens during the prepare phase.
    """
    import asyncio

    SessionLocal = _session_factory()

    with SessionLocal() as db:
        user = _create_user(db)
        _create_openai_provider_stack(db, user_id=user.id)
        rag_service = FakeGraphRagService(ingest_calls=[], retrieve_calls=[])

        graph_result = asyncio.run(
            run_chat_graph(
                ChatRequest(
                    prompt="",  # no text — attachment only
                    model_id="gpt-5-mini",
                    provider_code=ProviderCode.OPENAI,
                    attachments=[
                        ChatAttachment(
                            filename="report.pdf",
                            media_type="application/pdf",
                            url=_data_url("application/pdf", "dummy pdf content"),
                        )
                    ],
                ),
                db,
                user_id=user.id,
                rag_service=rag_service,
            )
        )

        # Ingestion called, retrieval deferred to tool call.
        assert len(rag_service.ingest_calls) == 1
        assert len(rag_service.retrieve_calls) == 0


