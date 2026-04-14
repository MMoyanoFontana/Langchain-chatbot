"""Thread memory and user fact extraction service.

Two responsibilities:
- get_memory_context: load thread summary + user facts for prompt injection
- update_memory: called after each assistant response to (a) extract user facts when
  the latest user message looks declarative and (b) regenerate the thread summary
  when enough messages have rolled out of the LLM history window
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatThread, MessageRole, UserMemory, utc_now

# Legacy threshold kept for tests that import it. The current path uses a rolling
# summary triggered by ROLLING_SUMMARY_DROP_TRIGGER instead.
SUMMARY_THRESHOLD = 20

# Regenerate the rolling thread summary when at least this many new messages have
# fallen out of the LLM history window since the last summary update.
ROLLING_SUMMARY_DROP_TRIGGER = 5

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

# Heuristic patterns that gate the LLM-based fact extractor. If the latest user
# message does not match any of these (Spanish + English biographical statements),
# we skip the extraction call entirely. Cheaper and avoids extracting noise like
# "preferred_format=table" from one-off formatting requests.
_DECLARATIVE_PATTERNS = re.compile(
    r"""
    \b(
        # English: identity / role / background
        i\s*am\b | i'?m\b | my\s+name\b | call\s+me\b |
        i\s+work\b | i\s+live\b | i\s+study\b | i\s+speak\b |
        i\s+have\s+\d+\s+years? | my\s+(role|job|profession|company|team|stack|language|background|expertise|degree)\b |
        # Spanish: identity / role / background
        soy\b | me\s+llamo | me\s+dedico | mi\s+nombre |
        trabajo\s+(en|de|como)\b | vivo\s+en\b | estudio\b | hablo\b |
        tengo\s+\d+\s+a[nñ]os | mi\s+(rol|trabajo|profesi[oó]n|empresa|equipo|stack|lenguaje|idioma|background|expertise|t[ií]tulo)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _looks_declarative(text: str | None) -> bool:
    """Cheap gate before invoking the LLM extractor."""
    if not text:
        return False
    return bool(_DECLARATIVE_PATTERNS.search(text))


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
    dropped_history_message_ids: list[str] | None = None,
) -> None:
    """Run the post-response memory updates.

    Two operations, both best-effort (any LLM or DB error is swallowed so a
    memory update failure can never break the chat response):

    1. **Fact extraction**: only when the latest user message looks declarative
       (gated by `_looks_declarative`). Avoids extracting noise from one-off
       formatting requests like "format as a markdown table".
    2. **Rolling summary**: when `dropped_history_message_ids` indicates that at
       least `ROLLING_SUMMARY_DROP_TRIGGER` new messages have fallen out of the
       LLM history window since the last summary update, regenerate the summary
       to cover everything that no longer fits in the live window.
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

    dropped_ids = dropped_history_message_ids or []
    if len(dropped_ids) - already_summarized >= ROLLING_SUMMARY_DROP_TRIGGER:
        await _regenerate_rolling_summary(
            db,
            thread_id=thread_id,
            dropped_message_ids=dropped_ids,
            llm=llm,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _extract_and_save_user_facts(
    db: Session,
    user_id: str,
    messages: list[ChatMessage],
    llm,
) -> None:
    """Extract stable biographical facts from the latest user message and upsert them.

    Skipped entirely when the latest user message does not look declarative —
    avoids spending an LLM call (and storing noise) for one-off requests like
    "format as a markdown table".
    """
    recent = messages[-_FACT_EXTRACTION_WINDOW:]

    last_user_msg: str | None = None
    for msg in recent:
        if msg.role == MessageRole.USER:
            last_user_msg = msg.content

    if last_user_msg is None or not _looks_declarative(last_user_msg):
        return

    prompt = (
        "Extract ONLY stable biographical or long-term facts about the user "
        "from this message. Allowed categories: name, profession/role, "
        "location, languages spoken, technical expertise, long-term goals.\n\n"
        "DO NOT extract:\n"
        "- Format preferences (markdown, tables, bullet points, code style)\n"
        "- One-off requests or commands (\"respond like X\", \"use Y format\")\n"
        "- Tone/style preferences for a single conversation\n"
        "- Topics of interest for a single conversation\n"
        "- Anything that is not a permanent personal attribute\n\n"
        "Return JSON only — no prose, no markdown fences.\n"
        'Format: {"facts": [{"key": "snake_case_key", "value": "fact value"}]}\n'
        'If nothing qualifies, return: {"facts": []}\n\n'
        f"User message: {last_user_msg[:600]}"
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


async def _regenerate_rolling_summary(
    db: Session,
    *,
    thread_id: str,
    dropped_message_ids: list[str],
    llm,
) -> None:
    """Build a rolling summary covering the messages that no longer fit in the LLM window.

    The summary is regenerated from scratch each time it fires (cheap and avoids
    drift) and is bounded by the dropped message set, so it never duplicates the
    content the LLM still sees in the live history window.
    """
    if not dropped_message_ids:
        return

    # Load the actual ChatMessage rows for the dropped IDs, in chronological order.
    dropped = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.id.in_(dropped_message_ids))
            .order_by(ChatMessage.created_at)
        )
    )
    if not dropped:
        return

    lines: list[str] = []
    for msg in dropped:
        role = "User" if msg.role == MessageRole.USER else "Assistant"
        lines.append(f"{role}: {(msg.content or '')[:500]}")
    conversation = "\n".join(lines)

    prompt = (
        "Summarize the following conversation excerpt concisely (3-6 sentences). "
        "Focus on key facts, decisions, names, and context the assistant needs to "
        "continue the conversation coherently. Do not invent details.\n\n"
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
                thread.summary_message_count = len(dropped_message_ids)
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
