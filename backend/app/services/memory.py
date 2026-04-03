"""Thread memory and user fact extraction service.

Two responsibilities:
- get_memory_context: load thread summary + user facts for prompt injection
- update_memory: called after each assistant response to extract user facts and
  (at threshold intervals) regenerate the thread summary
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatThread, MessageRole, UserMemory, utc_now

# Summarize when the thread reaches this many total messages (user + assistant).
# At 20 messages that's 10 exchanges — enough context to warrant a summary.
SUMMARY_THRESHOLD = 20

# How many recent messages to scan when extracting user facts.
_FACT_EXTRACTION_WINDOW = 4

# Cap how many user-memory facts are injected into each prompt.
_MAX_MEMORY_FACTS = 50

# Regex for rejecting fact keys/values that look like credentials.
_SENSITIVE_PATTERN = re.compile(
    r"(password|secret|token|api.?key|bearer|auth)",
    re.IGNORECASE,
)
_MAX_FACT_VALUE_LEN = 200


@dataclass
class MemoryContext:
    summary: str | None = None
    user_facts: list[dict[str, str]] = field(default_factory=list)


def get_memory_context(db: Session, thread_id: str, user_id: str) -> MemoryContext:
    """Load the thread summary and per-user fact store for prompt injection."""
    thread = db.get(ChatThread, thread_id)
    summary = (thread.summary or "").strip() or None if thread else None

    facts = list(
        db.scalars(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.updated_at.desc())
            .limit(_MAX_MEMORY_FACTS)
        )
    )
    user_facts = [{"key": f.key, "value": f.value} for f in facts]

    return MemoryContext(summary=summary, user_facts=user_facts)


async def update_memory(
    db: Session,
    thread_id: str,
    user_id: str,
    llm,
) -> None:
    """Extract user facts from the latest exchange and, at threshold, regenerate summary.

    Both operations are best-effort: any LLM or DB error is swallowed so that a
    memory update failure can never break the chat response.
    """
    thread = db.get(ChatThread, thread_id)
    if thread is None:
        return

    total_count = db.scalar(
        select(func.count(ChatMessage.id)).where(ChatMessage.thread_id == thread_id)
    ) or 0
    if total_count == 0:
        return

    # Load only the recent window needed for fact extraction.
    already_summarized = thread.summary_message_count or 0
    offset = max(0, already_summarized - _FACT_EXTRACTION_WINDOW)
    recent_messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at)
            .offset(offset)
        )
    )

    await _extract_and_save_user_facts(db, user_id, recent_messages, llm)

    if total_count >= SUMMARY_THRESHOLD and total_count % SUMMARY_THRESHOLD == 0:
        all_messages = list(
            db.scalars(
                select(ChatMessage)
                .where(ChatMessage.thread_id == thread_id)
                .order_by(ChatMessage.created_at)
            )
        )
        await _generate_and_save_summary(db, thread_id, all_messages, llm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _extract_and_save_user_facts(
    db: Session,
    user_id: str,
    messages: list[ChatMessage],
    llm,
) -> None:
    """Extract facts about the user from the most recent user message and upsert them."""
    recent = messages[-_FACT_EXTRACTION_WINDOW:]

    # Find the last user message in the window.
    last_user_msg: str | None = None
    for msg in recent:
        if msg.role == MessageRole.USER:
            last_user_msg = msg.content

    if last_user_msg is None:
        return

    prompt = (
        "Extract facts about the user from this message. "
        "Only include clearly stated facts (name, job, skills, preferences, goals, etc.). "
        "Return JSON only — no prose, no markdown fences.\n"
        'Format: {"facts": [{"key": "snake_case_key", "value": "fact value"}]}\n'
        'If nothing to extract, return: {"facts": []}\n\n'
        f"User: {last_user_msg[:600]}"
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = _extract_text(response.content)
        data = json.loads(_extract_json_object(raw))
        for fact in data.get("facts", []):
            key = str(fact.get("key", "")).strip()[:100]
            value = str(fact.get("value", "")).strip()[:_MAX_FACT_VALUE_LEN]
            if (
                key
                and value
                and not _SENSITIVE_PATTERN.search(key)
                and not _SENSITIVE_PATTERN.search(value)
            ):
                _upsert_user_memory(db, user_id, key, value)
        db.commit()
    except Exception:
        db.rollback()


async def _generate_and_save_summary(
    db: Session,
    thread_id: str,
    messages: list[ChatMessage],
    llm,
) -> None:
    """Summarize the full thread and save to ChatThread.summary."""
    lines: list[str] = []
    for msg in messages:
        role = "User" if msg.role == MessageRole.USER else "Assistant"
        lines.append(f"{role}: {msg.content[:500]}")
    conversation = "\n".join(lines)

    prompt = (
        "Summarize this conversation concisely (3-6 sentences). "
        "Focus on key topics, decisions, and context needed to continue the conversation.\n\n"
        f"{conversation}\n\n"
        "Summary:"
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = _extract_text(response.content).strip()
        if summary:
            thread = db.get(ChatThread, thread_id)
            if thread:
                thread.summary = summary
                thread.summary_message_count = len(messages)
                thread.updated_at = utc_now()
                db.commit()
    except Exception:
        db.rollback()


def _upsert_user_memory(db: Session, user_id: str, key: str, value: str) -> None:
    existing = db.scalar(
        select(UserMemory).where(
            UserMemory.user_id == user_id,
            UserMemory.key == key,
        )
    )
    if existing:
        existing.value = value
        existing.updated_at = utc_now()
    else:
        db.add(UserMemory(user_id=user_id, key=key, value=value))


def _extract_text(content: object) -> str:
    """Flatten LangChain message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _extract_json_object(text: str) -> str:
    """Return the first {...} block from text, stripping any surrounding prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text
