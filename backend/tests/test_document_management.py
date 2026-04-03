"""Tests for per-document delete and retry operations.

Covers:
- RagService.delete_document — Pinecone vector cleanup + DB row removal
- RagService.retry_document — re-ingestion from source message attachment
- DELETE /users/me/threads/{thread_id}/documents/{document_id}
- POST  /users/me/threads/{thread_id}/documents/{document_id}/retry
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
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
from app.services.pinecone_store import PineconeQueryMatch
from app.services.rag import (
    RagConfigurationError,
    RagService,
    RagSettings,
)
from app.services.current_user import require_current_user
from app.security import encrypt_secret


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

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
    encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_settings() -> RagSettings:
    return RagSettings(
        enabled=True,
        pinecone_api_key="fake",
        pinecone_index_name="fake",
        namespace_prefix="rag",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=None,
        chunk_size=1400,
        chunk_overlap=200,
        top_k=4,
        max_context_chars=5000,
        min_relevance_score=0.2,
        max_file_bytes=5 * 1024 * 1024,
    )


class FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1), 0.5] for i, _ in enumerate(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


@dataclass
class FakeVectorStore:
    upserts: list[tuple[str, list[object]]] = field(default_factory=list)
    deletes: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    query_results: list[PineconeQueryMatch] = field(default_factory=list)

    async def upsert(self, *, namespace: str, vectors: list[object]) -> None:
        self.upserts.append((namespace, vectors))

    async def query(self, *, namespace, query_vector, top_k, metadata_filter=None):
        return list(self.query_results)

    async def delete_by_filter(self, *, namespace: str, metadata_filter: dict) -> None:
        self.deletes.append((namespace, metadata_filter))

    async def describe_index_dimension(self) -> int:
        return 2


def _seed_db(db: Session) -> tuple[User, ChatThread, ChatMessage]:
    user = User(id="u1", email="u1@example.com", full_name="Test User", is_active=True)
    db.add(user)
    db.flush()

    provider = Provider(code=ProviderCode.OPENAI, display_name="OpenAI", is_active=True)
    db.add(provider)
    db.flush()
    db.add(ProviderModel(provider_id=provider.id, model_id="gpt-4o-mini", display_name="GPT-4o mini", is_active=True))
    db.add(ProviderApiKey(
        user_id=user.id,
        provider_id=provider.id,
        key_name="default",
        encrypted_api_key="unused",
        is_default=True,
        is_active=True,
    ))

    thread = ChatThread(id="t1", user_id=user.id, title="Test thread")
    db.add(thread)
    db.flush()

    content = "Hello world content for indexing."
    message = ChatMessage(
        id="m1",
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Test message",
        attachments=[
            {
                "type": "file",
                "filename": "test.txt",
                "media_type": "text/plain",
                "url": _data_url("text/plain", content),
            }
        ],
        citations=[],
        model_name="gpt-4o-mini",
        provider_id=provider.id,
    )
    db.add(message)
    db.commit()
    db.refresh(user)
    db.refresh(thread)
    db.refresh(message)
    return user, thread, message


def _make_indexed_document(
    db: Session,
    user: User,
    thread: ChatThread,
    message: ChatMessage,
    *,
    status: DocumentIndexStatus = DocumentIndexStatus.INDEXED,
    content: str = "Hello world content for indexing.",
) -> IndexedDocument:
    doc = IndexedDocument(
        id="doc1",
        user_id=user.id,
        thread_id=thread.id,
        source_message_id=message.id,
        filename="test.txt",
        media_type="text/plain",
        checksum_sha256=_checksum(content),
        byte_size=len(content.encode()),
        chunk_count=1 if status == DocumentIndexStatus.INDEXED else 0,
        pinecone_namespace=f"rag-{user.id}",
        status=status,
        error_message="Parse failed." if status == DocumentIndexStatus.FAILED else None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# RagService.delete_document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    def test_deletes_pinecone_vectors_and_db_row(self):
        factory = _session_factory()
        db = factory()
        vector_store = FakeVectorStore()
        service = RagService(settings=_make_settings(), vector_store=vector_store)

        user, thread, message = _seed_db(db)
        doc = _make_indexed_document(db, user, thread, message)

        asyncio.run(service.delete_document(
            db=db, user_id=user.id, thread_id=thread.id, document_id=doc.id
        ))

        # Pinecone delete called with correct filter
        assert len(vector_store.deletes) == 1
        namespace, filt = vector_store.deletes[0]
        assert namespace == doc.pinecone_namespace
        assert filt == {"document_id": {"$eq": doc.id}}

        # Row removed from DB
        assert db.get(IndexedDocument, doc.id) is None

    def test_skips_pinecone_for_failed_document(self):
        factory = _session_factory()
        db = factory()
        vector_store = FakeVectorStore()
        service = RagService(settings=_make_settings(), vector_store=vector_store)

        user, thread, message = _seed_db(db)
        doc = _make_indexed_document(db, user, thread, message, status=DocumentIndexStatus.FAILED)

        asyncio.run(service.delete_document(
            db=db, user_id=user.id, thread_id=thread.id, document_id=doc.id
        ))

        # No Pinecone call for a failed (never-indexed) document
        assert vector_store.deletes == []
        assert db.get(IndexedDocument, doc.id) is None

    def test_is_noop_for_unknown_document(self):
        factory = _session_factory()
        db = factory()
        vector_store = FakeVectorStore()
        service = RagService(settings=_make_settings(), vector_store=vector_store)

        user, thread, message = _seed_db(db)

        asyncio.run(service.delete_document(
            db=db, user_id=user.id, thread_id=thread.id, document_id="no-such-doc"
        ))

        assert vector_store.deletes == []

    def test_is_noop_when_rag_disabled(self):
        factory = _session_factory()
        db = factory()
        vector_store = FakeVectorStore()
        disabled_settings = RagSettings(
            enabled=False,
            pinecone_api_key=None,
            pinecone_index_name=None,
            namespace_prefix="rag",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=None,
            chunk_size=1400,
            chunk_overlap=200,
            top_k=4,
            max_context_chars=5000,
            min_relevance_score=0.2,
            max_file_bytes=5 * 1024 * 1024,
        )
        service = RagService(settings=disabled_settings, vector_store=None)

        user, thread, message = _seed_db(db)
        doc = _make_indexed_document(db, user, thread, message)

        asyncio.run(service.delete_document(
            db=db, user_id=user.id, thread_id=thread.id, document_id=doc.id
        ))

        # With RAG disabled, Pinecone is skipped but DB row is still removed
        assert vector_store.deletes == []
        assert db.get(IndexedDocument, doc.id) is None


# ---------------------------------------------------------------------------
# RagService.retry_document
# ---------------------------------------------------------------------------

class TestRetryDocument:
    def _make_service_with_fake_embeddings(self) -> tuple[RagService, FakeVectorStore]:
        vector_store = FakeVectorStore()
        service = RagService(settings=_make_settings(), vector_store=vector_store)
        # Patch embeddings client builder so we don't need a real OpenAI key
        async def _fake_embeddings(**_):
            return FakeEmbeddings()
        service._build_embeddings_client = _fake_embeddings  # type: ignore[method-assign]
        return service, vector_store

    def test_re_indexes_failed_document(self):
        factory = _session_factory()
        db = factory()
        service, vector_store = self._make_service_with_fake_embeddings()

        user, thread, message = _seed_db(db)
        doc = _make_indexed_document(db, user, thread, message, status=DocumentIndexStatus.FAILED)

        result = asyncio.run(service.retry_document(
            db=db, user_id=user.id, thread_id=thread.id, document_id=doc.id
        ))

        assert result.status == DocumentIndexStatus.INDEXED
        assert result.chunk_count > 0
        assert len(vector_store.upserts) == 1

    def test_raises_when_document_not_found(self):
        factory = _session_factory()
        db = factory()
        service, _ = self._make_service_with_fake_embeddings()
        user, thread, message = _seed_db(db)

        with pytest.raises(RagConfigurationError, match="not found"):
            asyncio.run(service.retry_document(
                db=db, user_id=user.id, thread_id=thread.id, document_id="no-such-doc"
            ))

    def test_raises_when_source_message_missing(self):
        factory = _session_factory()
        db = factory()
        service, _ = self._make_service_with_fake_embeddings()
        user, thread, message = _seed_db(db)

        doc = _make_indexed_document(db, user, thread, message, status=DocumentIndexStatus.FAILED)
        doc.source_message_id = None
        db.commit()

        with pytest.raises(RagConfigurationError, match="no source message"):
            asyncio.run(service.retry_document(
                db=db, user_id=user.id, thread_id=thread.id, document_id=doc.id
            ))

    def test_raises_when_no_matching_attachment(self):
        factory = _session_factory()
        db = factory()
        service, _ = self._make_service_with_fake_embeddings()
        user, thread, message = _seed_db(db)

        doc = _make_indexed_document(
            db, user, thread, message,
            status=DocumentIndexStatus.FAILED,
            content="different content that won't match",
        )

        with pytest.raises(RagConfigurationError, match="no matching attachment"):
            asyncio.run(service.retry_document(
                db=db, user_id=user.id, thread_id=thread.id, document_id=doc.id
            ))


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------

def _build_test_app(db: Session, current_user: User) -> TestClient:
    app = FastAPI()
    app.include_router(users_router.router)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[require_current_user] = lambda: current_user
    return TestClient(app, raise_server_exceptions=False)


class TestDeleteDocumentRoute:
    def test_returns_204_and_removes_document(self, monkeypatch):
        factory = _session_factory()
        db = factory()
        user, thread, message = _seed_db(db)
        doc = _make_indexed_document(db, user, thread, message)

        deleted_calls: list[tuple] = []

        async def fake_delete(*, db, user_id, thread_id, document_id):
            deleted_calls.append((user_id, thread_id, document_id))
            row = db.get(IndexedDocument, document_id)
            if row:
                db.delete(row)
                db.commit()

        monkeypatch.setattr(users_router.get_rag_service(), "delete_document", fake_delete)

        client = _build_test_app(db, user)
        response = client.delete(f"/users/me/threads/{thread.id}/documents/{doc.id}")

        assert response.status_code == 204
        assert len(deleted_calls) == 1
        assert db.get(IndexedDocument, doc.id) is None

    def test_returns_404_for_unknown_document(self):
        factory = _session_factory()
        db = factory()
        user, thread, message = _seed_db(db)

        client = _build_test_app(db, user)
        response = client.delete(f"/users/me/threads/{thread.id}/documents/no-such-doc")

        assert response.status_code == 404

    def test_returns_404_for_wrong_thread(self):
        factory = _session_factory()
        db = factory()
        user, thread, message = _seed_db(db)
        doc = _make_indexed_document(db, user, thread, message)

        client = _build_test_app(db, user)
        response = client.delete(f"/users/me/threads/wrong-thread/documents/{doc.id}")

        assert response.status_code == 404


class TestRetryDocumentRoute:
    def test_returns_422_for_non_failed_document(self):
        factory = _session_factory()
        db = factory()
        user, thread, message = _seed_db(db)
        # INDEXED document — retry not allowed
        _make_indexed_document(db, user, thread, message, status=DocumentIndexStatus.INDEXED)

        client = _build_test_app(db, user)
        response = client.post(f"/users/me/threads/{thread.id}/documents/doc1/retry")

        assert response.status_code == 422

    def test_returns_404_for_unknown_document(self):
        factory = _session_factory()
        db = factory()
        user, thread, message = _seed_db(db)

        client = _build_test_app(db, user)
        response = client.post(f"/users/me/threads/{thread.id}/documents/no-such-doc/retry")

        assert response.status_code == 404

    def test_returns_updated_document_on_success(self, monkeypatch):
        factory = _session_factory()
        db = factory()
        user, thread, message = _seed_db(db)
        doc = _make_indexed_document(db, user, thread, message, status=DocumentIndexStatus.FAILED)

        async def fake_retry(*, db, user_id, thread_id, document_id):
            row = db.get(IndexedDocument, document_id)
            row.status = DocumentIndexStatus.INDEXED
            row.chunk_count = 3
            row.error_message = None
            db.commit()
            db.refresh(row)
            return row

        monkeypatch.setattr(users_router.get_rag_service(), "retry_document", fake_retry)

        client = _build_test_app(db, user)
        response = client.post(f"/users/me/threads/{thread.id}/documents/{doc.id}/retry")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "indexed"
        assert body["chunk_count"] == 3
