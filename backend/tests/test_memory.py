"""Tests for the thread memory and user fact extraction service.

Covers:
- get_memory_context: returns summary + user facts from DB
- update_memory: fact extraction from latest exchange
- update_memory: summary generation at SUMMARY_THRESHOLD
- update_memory: no summary below threshold
- update_memory: upserts existing fact (updates value)
- _inject_memory_context helper
- _load_memory graph node
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.graphs._nodes import (
    ChatGraphState,
    _inject_memory_context,
    _load_memory,
)
from app.models import (
    ChatMessage,
    ChatThread,
    MessageRole,
    User,
    UserMemory,
    utc_now,
)
from app.services.memory import (
    SUMMARY_THRESHOLD,
    MemoryContext,
    get_memory_context,
    update_memory,
)


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


def _make_user(db: Session, email: str = "test@example.com") -> User:
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_thread(db: Session, user_id: str, summary: str | None = None) -> ChatThread:
    thread = ChatThread(user_id=user_id, title="Test", summary=summary)
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


def _add_message(
    db: Session,
    thread_id: str,
    role: MessageRole,
    content: str,
) -> ChatMessage:
    msg = ChatMessage(thread_id=thread_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

@dataclass
class _FakeAIMessage:
    content: str


class FakeLLM:
    """Synchronously records calls and returns canned responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls: list[str] = []
        self._responses = list(responses or [])
        self._index = 0

    async def ainvoke(self, messages: list[Any]) -> _FakeAIMessage:
        prompt = messages[-1].content if messages else ""
        self.calls.append(prompt)
        if self._index < len(self._responses):
            resp = self._responses[self._index]
        else:
            resp = '{"facts": []}'
        self._index += 1
        return _FakeAIMessage(content=resp)


# ---------------------------------------------------------------------------
# get_memory_context
# ---------------------------------------------------------------------------

class TestGetMemoryContext:
    def test_empty_when_no_summary_or_facts(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            ctx = get_memory_context(db, thread.id, user.id)
        assert ctx.summary is None
        assert ctx.user_facts == []

    def test_returns_thread_summary(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id, summary="We discussed Python async patterns.")
            ctx = get_memory_context(db, thread.id, user.id)
        assert ctx.summary == "We discussed Python async patterns."

    def test_returns_user_facts(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            db.add(UserMemory(user_id=user.id, key="name", value="Max"))
            db.add(UserMemory(user_id=user.id, key="language", value="Python"))
            db.commit()
            ctx = get_memory_context(db, thread.id, user.id)
        keys = {f["key"] for f in ctx.user_facts}
        assert keys == {"name", "language"}

    def test_returns_both_summary_and_facts(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id, summary="Summary text.")
            db.add(UserMemory(user_id=user.id, key="goal", value="get a job in AI"))
            db.commit()
            ctx = get_memory_context(db, thread.id, user.id)
        assert ctx.summary == "Summary text."
        assert any(f["key"] == "goal" for f in ctx.user_facts)


# ---------------------------------------------------------------------------
# update_memory — fact extraction
# ---------------------------------------------------------------------------

class TestUpdateMemoryFactExtraction:
    def test_extracts_and_saves_new_fact(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _add_message(db, thread.id, MessageRole.USER, "I work as a data scientist.")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "That's great!")

            llm = FakeLLM(responses=['{"facts": [{"key": "occupation", "value": "data scientist"}]}'])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            saved = db.scalar(select(UserMemory).where(UserMemory.user_id == user.id, UserMemory.key == "occupation"))
        assert saved is not None
        assert saved.value == "data scientist"

    def test_upserts_existing_fact(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            db.add(UserMemory(user_id=user.id, key="language", value="Java"))
            db.commit()

            _add_message(db, thread.id, MessageRole.USER, "I prefer Python now.")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Python is great!")

            llm = FakeLLM(responses=['{"facts": [{"key": "language", "value": "Python"}]}'])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            updated = db.scalar(select(UserMemory).where(UserMemory.user_id == user.id, UserMemory.key == "language"))
        assert updated.value == "Python"

    def test_skips_facts_with_empty_key_or_value(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _add_message(db, thread.id, MessageRole.USER, "Hello.")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Hi!")

            llm = FakeLLM(responses=['{"facts": [{"key": "", "value": "something"}, {"key": "name", "value": ""}]}'])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            count = len(list(db.scalars(select(UserMemory).where(UserMemory.user_id == user.id))))
        assert count == 0

    def test_no_crash_on_malformed_llm_response(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _add_message(db, thread.id, MessageRole.USER, "Hello.")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Hi!")

            llm = FakeLLM(responses=["this is not json"])
            # Should not raise
            asyncio.run(update_memory(db, thread.id, user.id, llm))


# ---------------------------------------------------------------------------
# update_memory — summary generation
# ---------------------------------------------------------------------------

class TestUpdateMemorySummary:
    def _fill_thread(self, db: Session, thread_id: str, n_pairs: int) -> None:
        for i in range(n_pairs):
            _add_message(db, thread_id, MessageRole.USER, f"User message {i}")
            _add_message(db, thread_id, MessageRole.ASSISTANT, f"Assistant reply {i}")

    def test_generates_summary_at_threshold(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            self._fill_thread(db, thread.id, SUMMARY_THRESHOLD // 2)  # exactly SUMMARY_THRESHOLD messages

            fact_resp = '{"facts": []}'
            summary_resp = "This is a test summary."
            # fact extraction is called once, then summary generation
            llm = FakeLLM(responses=[fact_resp, summary_resp])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            db.refresh(thread)
        assert thread.summary == "This is a test summary."
        assert thread.summary_message_count == SUMMARY_THRESHOLD

    def test_no_summary_below_threshold(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            # One pair = 2 messages, well below threshold
            _add_message(db, thread.id, MessageRole.USER, "Hello")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Hi")

            llm = FakeLLM(responses=['{"facts": []}'])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            db.refresh(thread)
        assert thread.summary is None
        # Only one call (fact extraction), no summary call
        assert len(llm.calls) == 1

    def test_no_summary_at_non_threshold_count(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            # SUMMARY_THRESHOLD + 2 messages (not a multiple of threshold)
            self._fill_thread(db, thread.id, SUMMARY_THRESHOLD // 2)
            _add_message(db, thread.id, MessageRole.USER, "Extra message")

            llm = FakeLLM(responses=['{"facts": []}'])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            db.refresh(thread)
        assert thread.summary is None


# ---------------------------------------------------------------------------
# _inject_memory_context helper
# ---------------------------------------------------------------------------

class TestInjectMemoryContext:
    def test_no_injection_when_no_memory(self):
        result = _inject_memory_context("User question", None, [])
        assert result == "User question"

    def test_prepends_summary(self):
        result = _inject_memory_context("User question", "Prior summary.", [])
        assert result.startswith("[Previous conversation summary]\nPrior summary.")
        assert "User question" in result

    def test_prepends_user_facts(self):
        facts = [{"key": "name", "value": "Max"}]
        result = _inject_memory_context("User question", None, facts)
        assert "[About the user]" in result
        assert "- name: Max" in result
        assert "User question" in result

    def test_prepends_both_summary_and_facts(self):
        facts = [{"key": "goal", "value": "AI engineer"}]
        result = _inject_memory_context("Prompt", "Summary.", facts)
        assert "[Previous conversation summary]" in result
        assert "[About the user]" in result
        assert "Prompt" in result

    def test_skips_facts_with_missing_key_or_value(self):
        facts = [{"key": "", "value": "something"}, {"key": "name", "value": ""}]
        result = _inject_memory_context("Prompt", None, facts)
        assert result == "Prompt"


# ---------------------------------------------------------------------------
# _load_memory graph node
# ---------------------------------------------------------------------------

class TestLoadMemoryNode:
    def test_loads_context_into_state(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id, summary="Some prior summary.")
            db.add(UserMemory(user_id=user.id, key="role", value="engineer"))
            db.commit()

            state: ChatGraphState = {"user_id": user.id, "thread_id": thread.id}
            config = {"configurable": {"db": db}}
            result = _load_memory(state, config)

        assert result["thread_summary"] == "Some prior summary."
        assert any(f["key"] == "role" for f in result["user_memory_facts"])

    def test_returns_empty_context_on_db_error(self):
        state: ChatGraphState = {"user_id": "x", "thread_id": "y"}
        # Pass a non-Session object to force an error in _get_db
        config = {"configurable": {"db": None}}
        result = _load_memory(state, config)
        assert result["thread_summary"] is None
        assert result["user_memory_facts"] == []
