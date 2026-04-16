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
    MemoryContext,
    _looks_declarative,
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
# _looks_declarative — gate before LLM extraction
# ---------------------------------------------------------------------------

class TestLooksDeclarative:
    """Cheap regex gate that decides whether to spend an LLM call extracting facts.

    Should match biographical statements (English + Spanish) and reject
    chitchat, format requests, and one-off commands."""

    def test_empty_or_none_is_not_declarative(self):
        assert _looks_declarative(None) is False
        assert _looks_declarative("") is False
        assert _looks_declarative("   ") is False

    def test_format_request_is_not_declarative(self):
        # The motivating bug: a single "format as a markdown table" request
        # used to land `preferred_format=table` in the user fact store.
        assert _looks_declarative("Please format the answer as a markdown table") is False
        assert _looks_declarative("Respond with bullet points") is False
        assert _looks_declarative("Use code blocks") is False

    def test_chitchat_is_not_declarative(self):
        assert _looks_declarative("Hello") is False
        assert _looks_declarative("Thanks!") is False
        assert _looks_declarative("What is the capital of France?") is False

    def test_english_identity_statements_match(self):
        assert _looks_declarative("I'm a data scientist") is True
        assert _looks_declarative("I am Max") is True
        assert _looks_declarative("My name is Maximiliano") is True
        assert _looks_declarative("Call me Max") is True

    def test_english_role_statements_match(self):
        assert _looks_declarative("I work at Anthropic") is True
        assert _looks_declarative("I live in Buenos Aires") is True
        assert _looks_declarative("I study machine learning") is True
        assert _looks_declarative("I have 10 years of experience in Python") is True

    def test_spanish_identity_statements_match(self):
        assert _looks_declarative("Soy ingeniero de software") is True
        assert _looks_declarative("Me llamo Max") is True
        assert _looks_declarative("Mi nombre es Maximiliano") is True

    def test_spanish_role_statements_match(self):
        assert _looks_declarative("Trabajo en una startup") is True
        assert _looks_declarative("Vivo en Argentina") is True
        assert _looks_declarative("Hablo español e inglés") is True
        assert _looks_declarative("Tengo 10 años de experiencia") is True

    def test_case_insensitive(self):
        assert _looks_declarative("SOY DESARROLLADOR") is True
        assert _looks_declarative("i AM a teacher") is True


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

            # Message must be declarative for the gated extractor to run.
            _add_message(db, thread.id, MessageRole.USER, "I work mostly in Python now.")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Python is great!")

            llm = FakeLLM(responses=['{"facts": [{"key": "language", "value": "Python"}]}'])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            updated = db.scalar(select(UserMemory).where(UserMemory.user_id == user.id, UserMemory.key == "language"))
        assert updated.value == "Python"


    def test_no_crash_on_malformed_llm_response(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            # Declarative so the extractor actually runs and consumes the response.
            _add_message(db, thread.id, MessageRole.USER, "I am a developer.")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Hi!")

            llm = FakeLLM(responses=["this is not json"])
            # Should not raise
            asyncio.run(update_memory(db, thread.id, user.id, llm))

    def test_skips_extractor_for_non_declarative_message(self):
        """The declarative gate must short-circuit before any LLM call.

        Motivated by the format-request bug: "format as a markdown table" used
        to invoke the extractor and land `preferred_format=table` in the store.
        """
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _add_message(db, thread.id, MessageRole.USER, "Format that as a markdown table")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Sure!")

            llm = FakeLLM(responses=['{"facts": [{"key": "preferred_format", "value": "table"}]}'])
            asyncio.run(update_memory(db, thread.id, user.id, llm))

            facts = list(db.scalars(select(UserMemory).where(UserMemory.user_id == user.id)))

        assert facts == []
        assert llm.calls == [], "Extractor should never have been invoked"


# ---------------------------------------------------------------------------
# update_memory — summary generation
# ---------------------------------------------------------------------------

class TestUpdateMemoryRollingSummary:
    """Rolling summary fires based on `dropped_history_message_ids`, not a fixed count."""

    def _fill_thread(self, db: Session, thread_id: str, n_pairs: int) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        for i in range(n_pairs):
            msgs.append(_add_message(db, thread_id, MessageRole.USER, f"User message {i}"))
            msgs.append(_add_message(db, thread_id, MessageRole.ASSISTANT, f"Assistant reply {i}"))
        return msgs

    def test_generates_summary_when_drop_trigger_reached(self):
        from app.services.memory import ROLLING_SUMMARY_DROP_TRIGGER

        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            msgs = self._fill_thread(db, thread.id, ROLLING_SUMMARY_DROP_TRIGGER)
            dropped_ids = [m.id for m in msgs]

            # Last user message is non-declarative, so fact extraction is skipped.
            # Only the summary call should land.
            llm = FakeLLM(responses=["This is a test summary."])
            asyncio.run(
                update_memory(
                    db,
                    thread.id,
                    user.id,
                    llm,
                    dropped_history_message_ids=dropped_ids,
                )
            )

            db.refresh(thread)
        assert thread.summary == "This is a test summary."
        assert thread.summary_message_count == len(dropped_ids)
        # Only one LLM call: the summary regeneration. Fact extractor was gated out.
        assert len(llm.calls) == 1

    def test_no_summary_when_no_dropped_messages(self):
        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            _add_message(db, thread.id, MessageRole.USER, "Hello")
            _add_message(db, thread.id, MessageRole.ASSISTANT, "Hi")

            llm = FakeLLM(responses=[])
            asyncio.run(
                update_memory(
                    db,
                    thread.id,
                    user.id,
                    llm,
                    dropped_history_message_ids=[],
                )
            )

            db.refresh(thread)
        assert thread.summary is None
        # No LLM calls: "Hello" is non-declarative AND nothing dropped.
        assert len(llm.calls) == 0

    def test_no_summary_below_drop_trigger(self):
        from app.services.memory import ROLLING_SUMMARY_DROP_TRIGGER

        SessionLocal = _session_factory()
        with SessionLocal() as db:
            user = _make_user(db)
            thread = _make_thread(db, user.id)
            msgs = self._fill_thread(db, thread.id, ROLLING_SUMMARY_DROP_TRIGGER - 1)
            dropped_ids = [m.id for m in msgs[: ROLLING_SUMMARY_DROP_TRIGGER - 1]]

            llm = FakeLLM(responses=[])
            asyncio.run(
                update_memory(
                    db,
                    thread.id,
                    user.id,
                    llm,
                    dropped_history_message_ids=dropped_ids,
                )
            )

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
