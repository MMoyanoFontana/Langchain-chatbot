"""Chat history loading: walks the branch lineage and selects a token-budgeted window.

The chat graph relies on this to feed the LLM real prior turns instead of just the
current message. Older messages that fall outside the window are reported back so the
rolling-summary path in `app.services.memory` can fold them into `ChatThread.summary`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from app.models import ChatMessage, MessageRole

# Conservative budget for the history window only — leaves room for system prompt,
# memory addendum, retrieved chunks, current user message, and tool overhead.
DEFAULT_HISTORY_TOKEN_BUDGET = 6000

# Approximate per-message structural overhead (role markers, separators).
_PER_MESSAGE_TOKEN_OVERHEAD = 4

# Lazily-initialized tiktoken encoding (cl100k_base is a reasonable approximation
# across providers; differences are <15%).
_encoding = None


def _count_tokens(text: str) -> int:
    global _encoding
    if _encoding is None:
        import tiktoken

        _encoding = tiktoken.get_encoding("cl100k_base")
    return len(_encoding.encode(text or ""))


@dataclass
class HistoryMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class HistoryWindow:
    messages: list[HistoryMessage] = field(default_factory=list)
    dropped_count: int = 0
    dropped_message_ids: list[str] = field(default_factory=list)


def _walk_parent_chain(db: Session, leaf_message_id: str | None) -> list[ChatMessage]:
    """Walk `parent_message_id` from leaf upward, returning the chain chronologically."""
    chain: list[ChatMessage] = []
    seen: set[str] = set()
    current_id = leaf_message_id
    while current_id and current_id not in seen:
        seen.add(current_id)
        msg = db.get(ChatMessage, current_id)
        if msg is None:
            break
        chain.append(msg)
        current_id = msg.parent_message_id
    chain.reverse()
    return chain


def _serialize_message_content(msg: ChatMessage) -> str:
    content = msg.content or ""
    attachments = msg.attachments or []
    if attachments:
        labels = []
        for attachment in attachments:
            label = attachment.get("filename") or attachment.get("media_type") or "Attachment"
            labels.append(label)
        marker = f"[Attached: {', '.join(labels)}]"
        content = f"{content}\n{marker}" if content else marker
    return content


def select_history_window(
    db: Session,
    *,
    leaf_message_id: str,
    token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
) -> HistoryWindow:
    """Build a chat history window for the LLM.

    `leaf_message_id` is the CURRENT user message (the one being answered). Its
    ancestors form the candidate history; the leaf itself is excluded because the
    streaming node sends it as the final HumanMessage. Tool/system messages are
    skipped.
    """
    full_chain = _walk_parent_chain(db, leaf_message_id)
    if len(full_chain) <= 1:
        return HistoryWindow()

    ancestors = [m for m in full_chain[:-1] if m.role in (MessageRole.USER, MessageRole.ASSISTANT)]

    selected: list[ChatMessage] = []
    used_tokens = 0
    for msg in reversed(ancestors):
        content = _serialize_message_content(msg)
        msg_tokens = _count_tokens(content) + _PER_MESSAGE_TOKEN_OVERHEAD
        if selected and used_tokens + msg_tokens > token_budget:
            break
        selected.append(msg)
        used_tokens += msg_tokens
    selected.reverse()

    selected_ids = {m.id for m in selected}
    dropped = [m for m in ancestors if m.id not in selected_ids]

    return HistoryWindow(
        messages=[
            HistoryMessage(
                role="user" if m.role == MessageRole.USER else "assistant",
                content=_serialize_message_content(m),
            )
            for m in selected
        ],
        dropped_count=len(dropped),
        dropped_message_ids=[m.id for m in dropped],
    )
