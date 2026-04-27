"""Tests for reasoning/thinking-model support.

Covers:
- _build_chat_client: passes provider-specific reasoning params when supports_reasoning=True
- _extract_chunk_reasoning: pulls thinking text out of Anthropic/Gemini/OpenAI chunk formats
- _stream_model_and_persist: reasoning deltas stream via \x00REASONING: sentinel and
  are persisted on the ChatMessage.reasoning_content column.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.graphs._nodes import ChatModelStreamState
from app.graphs._stream import (
    _ThinkTagParser,
    _build_chat_client,
    _extract_chunk_reasoning,
    _extract_chunk_text,
    _stream_model_and_persist,
)
from app.models import ChatMessage, ChatThread, MessageRole, ProviderCode, User


def _session_factory() -> sessionmaker:
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False, class_=Session
    )


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


@dataclass
class _FakeReasoningChunk:
    """Chunk that carries both thinking blocks and visible text."""

    content: Any  # list[dict] or str
    additional_kwargs: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)

    def __add__(self, other: "_FakeReasoningChunk") -> "_FakeReasoningChunk":
        # Content concatenation is only exercised when both sides are strings
        # or both are lists; the tests only feed one shape at a time.
        if isinstance(self.content, str) and isinstance(other.content, str):
            merged = self.content + other.content
        else:
            merged = list(self.content or []) + list(other.content or [])
        return _FakeReasoningChunk(
            content=merged,
            additional_kwargs={**self.additional_kwargs, **other.additional_kwargs},
            tool_calls=list(self.tool_calls) + list(other.tool_calls),
        )


class _ReasoningFakeLLM:
    """Streams a thinking chunk followed by answer text chunks."""

    def __init__(
        self,
        *,
        thinking_text: str = "Let me think...",
        answer_text: str = "Here is the answer.",
    ) -> None:
        self._thinking_text = thinking_text
        self._answer_text = answer_text
        self.stream_calls: list[list[Any]] = []

    async def astream(self, messages: list[Any]) -> AsyncIterator[_FakeReasoningChunk]:
        self.stream_calls.append(messages)
        yield _FakeReasoningChunk(
            content=[{"type": "thinking", "thinking": self._thinking_text}]
        )
        for ch in self._answer_text:
            yield _FakeReasoningChunk(content=[{"type": "text", "text": ch}])

    def bind_tools(self, tools: list) -> "_ReasoningFakeLLM":
        return self


def _make_state(user: User, thread: ChatThread, *, supports_reasoning: bool) -> ChatModelStreamState:
    return {
        "prompt": "What is 2+2?",
        "retrieved_chunks": [],
        "user_id": user.id,
        "thread_id": thread.id,
        "provider_id": 1,
        "provider_api_key_id": "key-1",
        "selected_provider_code": "anthropic",
        "selected_model_id": "claude-sonnet-4-20250514",
        "thread_has_documents": False,
        "supports_reasoning": supports_reasoning,
    }


def _run_stream(db: Session, state: ChatModelStreamState, llm: _ReasoningFakeLLM) -> list[str]:
    written: list[str] = []
    import app.graphs._stream as _stream_module

    original_build = _stream_module._build_chat_client
    original_resolve = _stream_module._resolve_provider_api_key_value
    original_get_writer = _stream_module.get_stream_writer

    _stream_module._build_chat_client = lambda **_kw: llm
    _stream_module._resolve_provider_api_key_value = lambda **_kw: "fake-key"
    _stream_module.get_stream_writer = lambda: written.append

    try:
        asyncio.run(_stream_model_and_persist(state, {"configurable": {"db": db}}))
    finally:
        _stream_module._build_chat_client = original_build
        _stream_module._resolve_provider_api_key_value = original_resolve
        _stream_module.get_stream_writer = original_get_writer

    return written


class TestExtractChunkReasoning:
    def test_anthropic_thinking_block(self):
        chunk = _FakeReasoningChunk(
            content=[{"type": "thinking", "thinking": "hmm... 2+2 is 4"}]
        )
        assert _extract_chunk_reasoning(chunk) == "hmm... 2+2 is 4"

    def test_gemini_thought_part(self):
        chunk = _FakeReasoningChunk(
            content=[{"thought": True, "text": "thinking step"}]
        )
        assert _extract_chunk_reasoning(chunk) == "thinking step"

    def test_openai_reasoning_summary(self):
        chunk = _FakeReasoningChunk(
            content=[],
            additional_kwargs={
                "reasoning": {"summary": [{"text": "s1"}, {"text": "s2"}]}
            },
        )
        assert _extract_chunk_reasoning(chunk) == "s1s2"

    def test_plain_text_chunk_has_no_reasoning(self):
        chunk = _FakeReasoningChunk(content=[{"type": "text", "text": "hello"}])
        assert _extract_chunk_reasoning(chunk) == ""

    def test_extract_chunk_text_skips_thinking_blocks(self):
        content = [
            {"type": "thinking", "thinking": "private"},
            {"type": "text", "text": "public"},
        ]
        assert _extract_chunk_text(content) == "public"


class TestBuildChatClient:
    def test_anthropic_reasoning_sets_thinking_param(self):
        client = _build_chat_client(
            provider_code=ProviderCode.ANTHROPIC,
            model_name="claude-opus-4-1-20250805",
            api_key="sk-ant-test",
            supports_reasoning=True,
        )
        assert getattr(client, "thinking", None) == {
            "type": "enabled",
            "budget_tokens": 4000,
        }
        assert getattr(client, "temperature", None) == 1

    def test_anthropic_without_reasoning_uses_default_temperature(self):
        client = _build_chat_client(
            provider_code=ProviderCode.ANTHROPIC,
            model_name="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            supports_reasoning=False,
        )
        assert getattr(client, "thinking", None) is None
        assert getattr(client, "temperature", None) == 0.2

    def test_openai_reasoning_sets_effort(self):
        client = _build_chat_client(
            provider_code=ProviderCode.OPENAI,
            model_name="gpt-5.2",
            api_key="sk-test",
            supports_reasoning=True,
        )
        assert getattr(client, "reasoning_effort", None) == "medium"


class TestStreamReasoning:
    def test_reasoning_delta_emitted_via_sentinel(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = _ReasoningFakeLLM(thinking_text="step 1 step 2", answer_text="ok")
            state = _make_state(user, thread, supports_reasoning=True)
            written = _run_stream(db, state, llm)

        reasoning_frames = [w for w in written if w.startswith("\x00REASONING:")]
        assert len(reasoning_frames) == 1
        payload = json.loads(reasoning_frames[0][len("\x00REASONING:"):])
        assert payload == "step 1 step 2"

    def test_reasoning_persisted_on_message(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = _ReasoningFakeLLM(thinking_text="internal note", answer_text="42")
            state = _make_state(user, thread, supports_reasoning=True)
            _run_stream(db, state, llm)

            saved = db.scalar(
                select(ChatMessage).where(
                    ChatMessage.thread_id == thread.id,
                    ChatMessage.role == MessageRole.ASSISTANT,
                )
            )
        assert saved is not None
        assert saved.reasoning_content == "internal note"
        assert saved.content == "42"

    def test_reasoning_skipped_when_unsupported(self):
        """With supports_reasoning=False, thinking blocks must not leak to the client."""
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = _ReasoningFakeLLM(thinking_text="should not leak", answer_text="hi")
            state = _make_state(user, thread, supports_reasoning=False)
            written = _run_stream(db, state, llm)

            saved = db.scalar(
                select(ChatMessage).where(
                    ChatMessage.thread_id == thread.id,
                    ChatMessage.role == MessageRole.ASSISTANT,
                )
            )

        assert not any(w.startswith("\x00REASONING:") for w in written)
        assert saved is not None
        assert saved.reasoning_content is None


class TestThinkTagParser:
    """Unit tests for the inline <think> tag parser used by Qwen-style models."""

    def test_simple_think_block(self):
        parser = _ThinkTagParser()
        vis, reason = parser.feed("<think>reasoning here</think>answer")
        assert reason == "reasoning here"
        assert vis == "answer"

    def test_think_block_across_chunks(self):
        parser = _ThinkTagParser()
        vis1, reason1 = parser.feed("<think>step 1")
        assert vis1 == ""
        assert reason1 == "step 1"

        vis2, reason2 = parser.feed(" step 2</think>")
        assert vis2 == ""
        assert reason2 == " step 2"

        vis3, reason3 = parser.feed("final answer")
        assert vis3 == "final answer"
        assert reason3 == ""

    def test_partial_open_tag_at_boundary(self):
        parser = _ThinkTagParser()
        # Feed text ending with a partial "<thi" — should hold back the partial
        vis1, reason1 = parser.feed("hello <thi")
        assert reason1 == ""
        assert vis1 == "hello "

        # Complete the tag
        vis2, reason2 = parser.feed("nk>inside</think>after")
        assert reason2 == "inside"
        assert vis2 == "after"

    def test_partial_close_tag_at_boundary(self):
        parser = _ThinkTagParser()
        parser.feed("<think>")
        vis, reason = parser.feed("data</thi")
        assert reason == "data"
        assert vis == ""

        vis2, reason2 = parser.feed("nk>visible")
        assert reason2 == ""
        assert vis2 == "visible"

    def test_no_think_tags(self):
        parser = _ThinkTagParser()
        vis, reason = parser.feed("just plain text")
        assert vis == "just plain text"
        assert reason == ""

    def test_flush_inside_think(self):
        """Flush emits buffered partial close-tag content as reasoning."""
        parser = _ThinkTagParser()
        parser.feed("<think>data</thi")  # "</thi" is a partial close tag
        vis, reason = parser.flush()
        # The partial "</thi" was held back; flush releases it as reasoning
        assert "thi" in reason or "</thi" in reason
        assert vis == ""

    def test_flush_outside_think(self):
        """Flush emits buffered partial open-tag content as visible text."""
        parser = _ThinkTagParser()
        parser.feed("hello <th")  # "<th" is a partial open tag
        vis, reason = parser.flush()
        assert "<th" in vis
        assert reason == ""

    def test_multiple_think_blocks(self):
        parser = _ThinkTagParser()
        vis, reason = parser.feed("<think>a</think>mid<think>b</think>end")
        assert reason == "ab"
        assert vis == "midend"

    def test_empty_think_block(self):
        parser = _ThinkTagParser()
        vis, reason = parser.feed("<think></think>answer")
        assert reason == ""
        assert vis == "answer"

    def test_newlines_in_think_block(self):
        """Qwen models typically output multi-line reasoning."""
        parser = _ThinkTagParser()
        vis, reason = parser.feed("<think>\nstep 1\nstep 2\n</think>\nThe answer is 4.")
        assert reason == "\nstep 1\nstep 2\n"
        assert vis == "\nThe answer is 4."


class _QwenThinkFakeLLM:
    """Simulates a Qwen QWQ model that outputs <think> tags inline."""

    def __init__(
        self,
        *,
        thinking_text: str = "Let me think...",
        answer_text: str = "The answer.",
    ) -> None:
        self._thinking_text = thinking_text
        self._answer_text = answer_text

    async def astream(self, messages: list[Any]) -> AsyncIterator[_FakeReasoningChunk]:
        # Qwen outputs reasoning as plain string content with <think> tags
        yield _FakeReasoningChunk(content=f"<think>{self._thinking_text}</think>")
        for ch in self._answer_text:
            yield _FakeReasoningChunk(content=ch)

    def bind_tools(self, tools: list) -> "_QwenThinkFakeLLM":
        return self


class TestStreamQwenThinkTags:
    def test_think_tags_routed_to_reasoning(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = _QwenThinkFakeLLM(thinking_text="step by step", answer_text="42")
            state = _make_state(user, thread, supports_reasoning=True)
            state["selected_provider_code"] = "groq"
            state["selected_model_id"] = "qwen/qwq-32b"
            written = _run_stream(db, state, llm)

        reasoning_frames = [w for w in written if w.startswith("\x00REASONING:")]
        assert len(reasoning_frames) >= 1
        combined = "".join(
            json.loads(f[len("\x00REASONING:"):]) for f in reasoning_frames
        )
        assert "step by step" in combined

        # The visible text must NOT contain <think> tags
        text_frames = [w for w in written if not w.startswith("\x00")]
        visible = "".join(text_frames)
        assert "<think>" not in visible
        assert "step by step" not in visible
        assert "42" in visible

    def test_think_tags_persisted_as_reasoning(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            llm = _QwenThinkFakeLLM(thinking_text="private thought", answer_text="public")
            state = _make_state(user, thread, supports_reasoning=True)
            state["selected_provider_code"] = "groq"
            state["selected_model_id"] = "qwen/qwq-32b"
            _run_stream(db, state, llm)

            saved = db.scalar(
                select(ChatMessage).where(
                    ChatMessage.thread_id == thread.id,
                    ChatMessage.role == MessageRole.ASSISTANT,
                )
            )
        assert saved is not None
        assert saved.reasoning_content == "private thought"
        assert saved.content == "public"
        assert "<think>" not in saved.content
