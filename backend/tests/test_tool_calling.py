"""Tests for LLM-driven tool calling (agentic RAG).

Covers:
- _retrieve_context: sets thread_has_documents=True only for INDEXED docs
- _format_tool_retrieval_result: formats chunks and notice correctly
- _stream_model_and_persist: tool call path — retrieval executed, chunks saved as citations
- _stream_model_and_persist: no-tool path — text streamed directly
- _stream_model_and_persist: no-documents path — tool never offered
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import pytest
from langchain_core.messages import AIMessageChunk
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.graphs._nodes import ChatModelStreamState, RetrievedChunkState
from app.graphs._stream import _format_tool_retrieval_result, _stream_model_and_persist
from app.models import (
    ChatMessage,
    ChatThread,
    DocumentIndexStatus,
    IndexedDocument,
    MessageRole,
    User,
    utc_now,
)
from app.services.rag import RetrievalResult, RetrievedChunk


# ---------------------------------------------------------------------------
# Test DB helpers
# ---------------------------------------------------------------------------

def _session_factory() -> sessionmaker:
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, class_=Session)


def _make_user(db: Session) -> User:
    user = User(email="test@example.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_thread(db: Session, user_id: str) -> ChatThread:
    thread = ChatThread(user_id=user_id, title="Test")
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


def _make_indexed_document(
    db: Session,
    user_id: str,
    thread_id: str,
    status: DocumentIndexStatus = DocumentIndexStatus.INDEXED,
) -> IndexedDocument:
    doc = IndexedDocument(
        user_id=user_id,
        thread_id=thread_id,
        media_type="application/pdf",
        checksum_sha256="abc123",
        byte_size=1024,
        pinecone_namespace="test-ns",
        status=status,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


# ---------------------------------------------------------------------------
# Fake streaming LLM
# ---------------------------------------------------------------------------

@dataclass
class _FakeAIMessageChunk:
    content: str
    tool_calls: list[dict] = field(default_factory=list)

    def __add__(self, other: "_FakeAIMessageChunk") -> "_FakeAIMessageChunk":
        merged_tool_calls = list(self.tool_calls) + [tc for tc in other.tool_calls if tc not in self.tool_calls]
        return _FakeAIMessageChunk(
            content=self.content + other.content,
            tool_calls=merged_tool_calls,
        )


class FakeStreamingLLM:
    """Yields canned AIMessageChunks; optionally emits a tool call on first call."""

    def __init__(
        self,
        *,
        tool_call: dict | None = None,
        text: str = "Hello!",
        second_text: str = "Answer with context.",
    ) -> None:
        self._tool_call = tool_call
        self._text = text
        self._second_text = second_text
        self.stream_calls: list[list[Any]] = []
        self._call_count = 0

    async def astream(self, messages: list[Any]) -> AsyncIterator[_FakeAIMessageChunk]:
        self.stream_calls.append(messages)
        self._call_count += 1

        if self._call_count == 1 and self._tool_call is not None:
            # First call: emit tool call chunk (no text content)
            yield _FakeAIMessageChunk(content="", tool_calls=[self._tool_call])
        elif self._call_count == 1:
            # First call, no tool: stream text
            for char in self._text:
                yield _FakeAIMessageChunk(content=char)
        else:
            # Second call (after tool execution): stream final answer
            for char in self._second_text:
                yield _FakeAIMessageChunk(content=char)

    def bind_tools(self, tools: list) -> "FakeStreamingLLM":
        return self


# ---------------------------------------------------------------------------
# Fake RAG service
# ---------------------------------------------------------------------------

@dataclass
class FakeRetrievalResult:
    chunks: tuple
    thread_has_documents: bool = True
    notice: str | None = None


class FakeRagService:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.retrieve_calls: list[dict] = []
        self._chunks = tuple(chunks or [])

    async def retrieve(self, **kwargs) -> RetrievalResult:
        self.retrieve_calls.append(kwargs)
        return RetrievalResult(thread_has_documents=True, chunks=self._chunks)


# ---------------------------------------------------------------------------
# _format_tool_retrieval_result
# ---------------------------------------------------------------------------

class TestFormatToolRetrievalResult:
    def test_no_chunks_returns_notice(self):
        result = RetrievalResult(thread_has_documents=True, notice="Nothing found.")
        assert _format_tool_retrieval_result(result) == "Nothing found."

    def test_no_chunks_no_notice_returns_default(self):
        result = RetrievalResult(thread_has_documents=True)
        assert "No relevant content" in _format_tool_retrieval_result(result)

    def test_formats_chunks_with_labels(self):
        chunks = (
            RetrievedChunk(document_id="doc1", filename="report.pdf", chunk_index=0, score=0.9, text="Revenue was $1M."),
            RetrievedChunk(document_id="doc1", filename="report.pdf", chunk_index=1, score=0.8, text="Costs were $0.5M."),
        )
        result = RetrievalResult(thread_has_documents=True, chunks=chunks)
        formatted = _format_tool_retrieval_result(result)
        assert "[1] report.pdf (chunk 0):" in formatted
        assert "Revenue was $1M." in formatted
        assert "[2] report.pdf (chunk 1):" in formatted
        assert "Costs were $0.5M." in formatted

    def test_formats_chunk_without_filename(self):
        chunks = (
            RetrievedChunk(document_id="doc-xyz", filename=None, chunk_index=2, score=0.7, text="Some text."),
        )
        result = RetrievalResult(thread_has_documents=True, chunks=chunks)
        formatted = _format_tool_retrieval_result(result)
        assert "Document doc-xyz" in formatted
        assert "Some text." in formatted


# ---------------------------------------------------------------------------
# _retrieve_context — thread_has_documents flag
# ---------------------------------------------------------------------------

class TestRetrieveContextDocCheck:
    """Tests that _retrieve_context sets thread_has_documents based on DB state."""

    def _run_retrieve_context(self, db: Session, user_id: str, thread_id: str) -> dict:
        from app.graphs._nodes import _retrieve_context
        from unittest.mock import MagicMock

        fake_rag = MagicMock()
        fake_rag.build_augmented_prompt.return_value = "base prompt"

        state = {
            "user_id": user_id,
            "thread_id": thread_id,
            "base_model_prompt": "hello",
            "ingestion_notices": [],
            "thread_summary": None,
            "user_memory_facts": [],
        }
        config = {"configurable": {"db": db, "rag_service": fake_rag}}
        return asyncio.run(_retrieve_context(state, config))

    def test_no_docs_sets_false(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            result = self._run_retrieve_context(db, user.id, thread.id)
        assert result["thread_has_documents"] is False

    def test_indexed_doc_sets_true(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _make_indexed_document(db, user.id, thread.id, DocumentIndexStatus.INDEXED)
            result = self._run_retrieve_context(db, user.id, thread.id)
        assert result["thread_has_documents"] is True

    def test_pending_doc_does_not_count(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _make_indexed_document(db, user.id, thread.id, DocumentIndexStatus.PENDING)
            result = self._run_retrieve_context(db, user.id, thread.id)
        assert result["thread_has_documents"] is False

    def test_failed_doc_does_not_count(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _make_indexed_document(db, user.id, thread.id, DocumentIndexStatus.FAILED)
            result = self._run_retrieve_context(db, user.id, thread.id)
        assert result["thread_has_documents"] is False

    def test_retrieve_context_clears_retrieved_chunks(self):
        """retrieved_chunks must be empty — tool call populates it later."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            result = self._run_retrieve_context(db, user.id, thread.id)
        assert result["retrieved_chunks"] == []


# ---------------------------------------------------------------------------
# _stream_model_and_persist — tool calling paths
# ---------------------------------------------------------------------------

def _make_stream_state(
    db: Session,
    user: User,
    thread: ChatThread,
    *,
    thread_has_documents: bool = False,
) -> ChatModelStreamState:
    return {
        "prompt": "What does the report say?",
        "retrieved_chunks": [],
        "user_id": user.id,
        "thread_id": thread.id,
        "provider_id": 1,
        "provider_api_key_id": "key-1",
        "selected_provider_code": "openai",
        "selected_model_id": "gpt-4o",
        "thread_has_documents": thread_has_documents,
    }


class TestStreamModelToolPath:
    def _run(
        self,
        db: Session,
        state: ChatModelStreamState,
        llm: FakeStreamingLLM,
        rag_service: FakeRagService | None = None,
    ) -> list[str]:
        written: list[str] = []
        import app.graphs._stream as _stream_module

        original_build = _stream_module._build_chat_client
        original_resolve = _stream_module._resolve_provider_api_key_value
        original_get_writer = _stream_module.get_stream_writer

        _stream_module._build_chat_client = lambda **_kw: llm
        _stream_module._resolve_provider_api_key_value = lambda **_kw: "fake-key"
        _stream_module.get_stream_writer = lambda: written.append

        try:
            config: dict = {"configurable": {"db": db}}
            if rag_service is not None:
                config["configurable"]["rag_service"] = rag_service
            asyncio.run(_stream_model_and_persist(state, config))
        finally:
            _stream_module._build_chat_client = original_build
            _stream_module._resolve_provider_api_key_value = original_resolve
            _stream_module.get_stream_writer = original_get_writer

        return written

    def test_no_tool_call_streams_text_in_first_pass(self):
        """When model responds with text (no tool call), text is streamed immediately
        in one pass regardless of thread_has_documents."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = FakeStreamingLLM(text="Direct answer.")
            state = _make_stream_state(db, user, thread, thread_has_documents=False)
            written = self._run(db, state, llm)

        assert "".join(written) == "Direct answer."
        assert llm._call_count == 1

    def test_no_tool_call_streams_text_directly(self):
        """When model responds with text (no tool call), text is streamed immediately."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = FakeStreamingLLM(tool_call=None, text="No need to search.")
            state = _make_stream_state(db, user, thread, thread_has_documents=True)
            written = self._run(db, state, llm)

        assert "".join(written) == "No need to search."
        assert llm._call_count == 1  # only first stream

    def test_tool_call_triggers_retrieval(self):
        """When model emits a tool call, retrieval is executed."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            tool_call = {"id": "call_1", "name": "search_documents", "args": {"query": "revenue figures"}}
            llm = FakeStreamingLLM(tool_call=tool_call, second_text="Revenue was $1M.")
            rag = FakeRagService()
            state = _make_stream_state(db, user, thread, thread_has_documents=True)
            self._run(db, state, llm, rag_service=rag)

        assert len(rag.retrieve_calls) == 1
        assert rag.retrieve_calls[0]["prompt"] == "revenue figures"

    def test_tool_call_streams_final_answer(self):
        """Final answer (second LLM call) is streamed to the client."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            tool_call = {"id": "call_1", "name": "search_documents", "args": {"query": "costs"}}
            llm = FakeStreamingLLM(tool_call=tool_call, second_text="Costs were $0.5M.")
            rag = FakeRagService()
            state = _make_stream_state(db, user, thread, thread_has_documents=True)
            written = self._run(db, state, llm, rag_service=rag)

        assert "".join(written) == "Costs were $0.5M."
        assert llm._call_count == 2  # first (tool call) + second (final answer)

    def test_tool_call_chunks_used_as_citations(self):
        """Retrieved chunks from the tool call are saved as message citations."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            chunks = (
                RetrievedChunk(document_id="doc1", filename="report.pdf", chunk_index=0, score=0.9, text="Revenue $1M."),
            )
            tool_call = {"id": "call_1", "name": "search_documents", "args": {"query": "revenue"}}
            llm = FakeStreamingLLM(tool_call=tool_call, second_text="Revenue was $1M.")
            rag = FakeRagService(chunks=list(chunks))
            state = _make_stream_state(db, user, thread, thread_has_documents=True)
            self._run(db, state, llm, rag_service=rag)

            saved_msg = db.scalar(
                select(ChatMessage).where(
                    ChatMessage.thread_id == thread.id,
                    ChatMessage.role == MessageRole.ASSISTANT,
                )
            )

        assert saved_msg is not None
        assert len(saved_msg.citations) == 1
        assert saved_msg.citations[0]["document_id"] == "doc1"
        assert saved_msg.citations[0]["filename"] == "report.pdf"

    def test_no_tool_call_saves_empty_citations(self):
        """When no tool call, citations are empty."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = FakeStreamingLLM(text="Just answering.")
            state = _make_stream_state(db, user, thread, thread_has_documents=True)
            self._run(db, state, llm)

            saved_msg = db.scalar(
                select(ChatMessage).where(
                    ChatMessage.thread_id == thread.id,
                    ChatMessage.role == MessageRole.ASSISTANT,
                )
            )

        assert saved_msg is not None
        assert saved_msg.citations == []


# ---------------------------------------------------------------------------
# _execute_tool_call dispatcher
# ---------------------------------------------------------------------------

class TestExecuteToolCall:
    def _run(self, tool_call: dict, state: ChatModelStreamState, config: dict) -> tuple[str, list]:
        from app.graphs._stream import _execute_tool_call
        return asyncio.run(_execute_tool_call(tool_call, state, config))

    def _base_state(self) -> ChatModelStreamState:
        return {
            "prompt": "fallback query",
            "retrieved_chunks": [],
            "user_id": "u1",
            "thread_id": "t1",
            "provider_id": 1,
            "provider_api_key_id": "key-1",
            "selected_provider_code": "openai",
            "selected_model_id": "gpt-4o",
            "thread_has_documents": False,
        }

    def test_unknown_tool_returns_error_message(self):
        content, chunks = self._run(
            {"name": "nonexistent_tool", "args": {}},
            self._base_state(),
            {"configurable": {}},
        )
        assert "Unknown tool" in content
        assert chunks == []
