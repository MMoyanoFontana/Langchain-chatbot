from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import HTTPException, status
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from app.models import ProviderCode
from app.services.memory import update_memory
from app.services.rag import RetrievalResult

from app.graphs._nodes import (
    ASSISTANT_SYSTEM_PROMPT,
    ChatModelStreamState,
    RetrievedChunkState,
    _get_db,
    _get_rag_service,
    _persist_assistant_message,
    _persist_tool_message,
    _resolve_provider_api_key_value,
)


# ---------------------------------------------------------------------------
# LLM client construction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inline <think> tag parser for models that embed reasoning in plain text
# (e.g. Qwen QWQ on Groq).  Stateful across chunks so partial tags at
# chunk boundaries are handled correctly.
# ---------------------------------------------------------------------------

class _ThinkTagParser:
    """Separates ``<think>…</think>`` blocks from regular text in a stream."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Feed a chunk.  Returns ``(visible_text, reasoning_text)``."""
        self._buf += text
        visible: list[str] = []
        reasoning: list[str] = []

        while True:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx >= 0:
                    reasoning.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self._CLOSE) :]
                    self._in_think = False
                    continue
                held = self._held_suffix(self._CLOSE)
                emit_up_to = len(self._buf) - held
                if emit_up_to > 0:
                    reasoning.append(self._buf[:emit_up_to])
                    self._buf = self._buf[emit_up_to:]
                break
            else:
                idx = self._buf.find(self._OPEN)
                if idx >= 0:
                    visible.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self._OPEN) :]
                    self._in_think = True
                    continue
                held = self._held_suffix(self._OPEN)
                emit_up_to = len(self._buf) - held
                if emit_up_to > 0:
                    visible.append(self._buf[:emit_up_to])
                    self._buf = self._buf[emit_up_to:]
                break

        return "".join(visible), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        """Flush remaining buffer (call after stream ends)."""
        remaining = self._buf
        self._buf = ""
        if self._in_think:
            return "", remaining
        return remaining, ""

    # -- internal --

    def _held_suffix(self, tag: str) -> int:
        """Length of the longest suffix of ``self._buf`` that is a prefix of *tag*."""
        max_check = min(len(tag) - 1, len(self._buf))
        for n in range(max_check, 0, -1):
            if self._buf.endswith(tag[:n]):
                return n
        return 0


def _extract_chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                block_type = item.get("type")
                # Skip thinking/reasoning blocks — they are streamed separately
                # via _extract_chunk_reasoning so the frontend can render them
                # in a collapsible reasoning panel.
                if block_type in {"thinking", "reasoning"}:
                    continue
                if item.get("thought") is True:
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)
    return ""


def _extract_chunk_reasoning(chunk: object) -> str:
    """Pull thinking/reasoning text out of a LangChain stream chunk.

    Handles Anthropic `thinking` blocks and Gemini `thought` parts carried in
    `content`, plus OpenAI reasoning summaries surfaced under
    `additional_kwargs["reasoning"]`.
    """
    parts: list[str] = []
    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            block_type = item.get("type")
            if block_type == "thinking":
                thinking = item.get("thinking")
                if isinstance(thinking, str):
                    parts.append(thinking)
            elif block_type == "reasoning":
                reasoning = item.get("reasoning") or item.get("text")
                if isinstance(reasoning, str):
                    parts.append(reasoning)
            elif item.get("thought") is True:
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)

    extra = getattr(chunk, "additional_kwargs", None) or {}
    reasoning_blob = extra.get("reasoning")
    if isinstance(reasoning_blob, dict):
        # OpenAI Responses-API format: { "summary": [{"text": "..."}, ...] }
        summary = reasoning_blob.get("summary")
        if isinstance(summary, list):
            for item in summary:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
        elif isinstance(reasoning_blob.get("text"), str):
            parts.append(reasoning_blob["text"])
    elif isinstance(reasoning_blob, str):
        parts.append(reasoning_blob)

    return "".join(parts)


ANTHROPIC_THINKING_BUDGET_TOKENS = 4000
OPENAI_REASONING_EFFORT = "medium"


def _build_chat_client(
    provider_code: ProviderCode,
    model_name: str,
    api_key: str,
    supports_reasoning: bool = False,
):
    if provider_code == ProviderCode.OPENAI:
        kwargs: dict[str, object] = {
            "model": model_name,
            "streaming": True,
            "stream_usage": True,
            "api_key": api_key,
        }
        if supports_reasoning:
            # Reasoning-capable OpenAI models (o-series, GPT-5.x) reject the
            # temperature parameter; control depth via reasoning_effort instead.
            kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT
        else:
            kwargs["temperature"] = 0.2
        return ChatOpenAI(**kwargs)

    if provider_code == ProviderCode.GROQ:
        return ChatOpenAI(
            model=model_name,
            streaming=True,
            stream_usage=True,
            temperature=0.2,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

    if provider_code == ProviderCode.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Anthropic chat provider is configured but `langchain-anthropic` is not installed. "
                    "Install it in the backend environment."
                ),
            ) from exc
        kwargs: dict[str, object] = {
            "model": model_name,
            "streaming": True,
            "api_key": api_key,
        }
        if supports_reasoning:
            # Extended thinking requires temperature=1 and a larger max_tokens
            # to accommodate both the thinking budget and the final answer.
            kwargs["temperature"] = 1
            kwargs["max_tokens"] = ANTHROPIC_THINKING_BUDGET_TOKENS + 4096
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGET_TOKENS,
            }
        else:
            kwargs["temperature"] = 0.2
        return ChatAnthropic(**kwargs)

    if provider_code == ProviderCode.GEMINI:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Gemini chat provider is configured but `langchain-google-genai` is not installed. "
                    "Install it in the backend environment."
                ),
            ) from exc
        kwargs: dict[str, object] = {
            "model": model_name,
            "temperature": 0.2,
            "google_api_key": api_key,
        }
        if supports_reasoning:
            # Expose Gemini's thought summaries in streamed chunks so we can
            # forward them to the client.
            kwargs["thinking_budget"] = -1  # dynamic: model picks its own budget
            kwargs["include_thoughts"] = True
        return ChatGoogleGenerativeAI(**kwargs)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Provider `{provider_code.value}` is not supported for chat.",
    )


# ---------------------------------------------------------------------------
# Tool definitions and execution
# ---------------------------------------------------------------------------

@tool
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for information relevant to the query.
    Call this whenever the user asks about content from files they uploaded in this conversation."""
    return ""  # Schema only; executed manually in _execute_tool_call


def _format_tool_retrieval_result(retrieval_result: RetrievalResult) -> str:
    if not retrieval_result.chunks:
        return retrieval_result.notice or "No relevant content found in uploaded documents."
    lines: list[str] = []
    for i, chunk in enumerate(retrieval_result.chunks, 1):
        label = chunk.filename or f"Document {chunk.document_id}"
        lines.append(f"[{i}] {label} (chunk {chunk.chunk_index}):\n{chunk.text}")
    return "\n\n".join(lines)


async def _execute_tool_call(
    tool_call: dict,
    state: ChatModelStreamState,
    config: RunnableConfig,
) -> tuple[str, list[RetrievedChunkState]]:
    """Dispatch a tool call and return (tool_message_content, retrieved_chunks)."""
    name = tool_call.get("name", "")
    args = tool_call.get("args") or {}

    if name == "search_documents":
        query = args.get("query") or state["prompt"]
        rag_service = _get_rag_service(config)
        retrieval_result = await rag_service.retrieve(
            db=_get_db(config),
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            prompt=query,
            preferred_provider_api_key_id=state["provider_api_key_id"],
        )
        chunks: list[RetrievedChunkState] = [
            {
                "document_id": c.document_id,
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "text": c.text,
            }
            for c in retrieval_result.chunks
        ]
        return _format_tool_retrieval_result(retrieval_result), chunks

    return f"Unknown tool: {name}", []


# ---------------------------------------------------------------------------
# Streaming node
# ---------------------------------------------------------------------------

def _extract_usage(chunk: AIMessageChunk | None) -> dict[str, int] | None:
    if chunk is None:
        return None
    usage = getattr(chunk, "usage_metadata", None)
    if not usage:
        return None
    try:
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(
                usage.get("total_tokens")
                or (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
            ),
        }
    except (TypeError, ValueError):
        return None


def _merge_usage(
    a: dict[str, int] | None, b: dict[str, int] | None
) -> dict[str, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    return {
        "input_tokens": a.get("input_tokens", 0) + b.get("input_tokens", 0),
        "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
    }


async def _stream_model_and_persist(
    state: ChatModelStreamState,
    config: RunnableConfig,
) -> dict[str, str]:
    db = _get_db(config)
    writer = get_stream_writer()
    collected_parts: list[str] = []
    collected_reasoning_parts: list[str] = []
    retrieved_chunks: list[RetrievedChunkState] = []
    started_at = time.perf_counter()
    first_token_at: float | None = None
    usage_total: dict[str, int] | None = None
    supports_reasoning = bool(state.get("supports_reasoning"))
    # Always parse <think> tags to strip them from visible content even if the
    # model isn't flagged supports_reasoning in the catalog.  Reasoning deltas
    # are only forwarded to the client via the REASONING sentinel when the model
    # is known to support reasoning (controls the UI panel).
    think_tag_parser = _ThinkTagParser()

    def _handle_chunk(chunk) -> str | None:
        """Forward reasoning/text deltas to the client.

        Returns the visible text delta so the caller can track first-token
        timing. Reasoning deltas are emitted via the REASONING sentinel and
        collected separately for persistence.
        """
        if supports_reasoning:
            reasoning_delta = _extract_chunk_reasoning(chunk)
            if reasoning_delta:
                collected_reasoning_parts.append(reasoning_delta)
                writer("\x00REASONING:" + json.dumps(reasoning_delta))
        chunk_text = _extract_chunk_text(chunk.content)
        # Parse inline <think> tags (Qwen-style models embed reasoning in
        # the plain text stream rather than using structured blocks).
        if chunk_text:
            visible, thinking = think_tag_parser.feed(chunk_text)
            if thinking:
                collected_reasoning_parts.append(thinking)
                if supports_reasoning:
                    writer("\x00REASONING:" + json.dumps(thinking))
            chunk_text = visible
        if chunk_text:
            collected_parts.append(chunk_text)
            writer(chunk_text)
            return chunk_text
        return None

    try:
        provider_code = ProviderCode(state["selected_provider_code"])
        api_key = _resolve_provider_api_key_value(
            db=db,
            user_id=state["user_id"],
            provider_id=state["provider_id"],
            provider_code=provider_code,
            provider_api_key_id=state["provider_api_key_id"],
        )
        llm = _build_chat_client(
            provider_code=provider_code,
            model_name=state["selected_model_id"],
            api_key=api_key,
            supports_reasoning=supports_reasoning,
        )

        thread_system_prompt = (state.get("thread_system_prompt") or "").strip()
        base_prompt = thread_system_prompt or ASSISTANT_SYSTEM_PROMPT
        system_addendum = state.get("system_addendum") or ""
        system_content = f"{base_prompt}\n\n{system_addendum}" if system_addendum else base_prompt
        messages: list = [SystemMessage(content=system_content)]
        for history_msg in state.get("history_messages") or []:
            role = history_msg.get("role")
            content = history_msg.get("content") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=state["prompt"]))

        # Offer search_documents when the thread has indexed documents.
        # Stream first response — text arrives immediately if no tool is called;
        # tool-call chunks have no content so nothing is written.
        # Fall back to plain streaming when the model doesn't support tool calling.
        tools = []
        if state.get("thread_has_documents"):
            tools.append(search_documents)
        llm_with_tools = llm.bind_tools(tools)
        accumulated: AIMessageChunk | None = None
        tools_supported = True
        messages_for_llm = list(messages)

        try:
            async for chunk in llm_with_tools.astream(messages_for_llm):
                if _handle_chunk(chunk) is not None and first_token_at is None:
                    first_token_at = time.perf_counter()
                accumulated = chunk if accumulated is None else accumulated + chunk  # type: ignore[operator]
        except (NotImplementedError, AttributeError):
            # Model doesn't support tool calling — fall back to plain streaming,
            # but only if we haven't already emitted any tokens to the client.
            if accumulated is not None or collected_parts:
                raise
            tools_supported = False
            collected_parts = []
            collected_reasoning_parts.clear()
            accumulated = None
            async for chunk in llm.astream(messages_for_llm):
                if _handle_chunk(chunk) is not None and first_token_at is None:
                    first_token_at = time.perf_counter()
                accumulated = chunk if accumulated is None else accumulated + chunk  # type: ignore[operator]
        usage_total = _merge_usage(usage_total, _extract_usage(accumulated))

        # Agentic tool-call loop: keep executing tool calls and re-streaming until
        # the model produces a final text response with no further tool calls.
        while tools_supported and accumulated is not None and accumulated.tool_calls:
            # Emit a visibility event for each tool call before executing.
            for tc in accumulated.tool_calls:
                query = (tc.get("args") or {}).get("query") or ""
                writer("\x00TOOL_CALL:" + json.dumps({"name": tc.get("name", ""), "query": query}))

            # Execute all tool calls in this turn in parallel.
            results = await asyncio.gather(
                *[_execute_tool_call(tc, state, config) for tc in accumulated.tool_calls]
            )

            tool_messages_list: list[ToolMessage] = []
            for tc, (tool_content, chunks) in zip(accumulated.tool_calls, results):
                retrieved_chunks.extend(chunks)
                _persist_tool_message(
                    db=db,
                    thread_id=state["thread_id"],
                    tool_name=tc.get("name", ""),
                    tool_input=tc.get("args") or {},
                    tool_output=tool_content,
                    model_name=state["selected_model_id"],
                    provider_id=state["provider_id"],
                )
                tool_messages_list.append(ToolMessage(content=tool_content, tool_call_id=tc["id"]))

            messages_for_llm = messages_for_llm + [accumulated] + tool_messages_list
            collected_parts = []
            accumulated = None

            async for chunk in llm_with_tools.astream(messages_for_llm):
                if _handle_chunk(chunk) is not None and first_token_at is None:
                    first_token_at = time.perf_counter()
                accumulated = chunk if accumulated is None else accumulated + chunk  # type: ignore[operator]
            usage_total = _merge_usage(usage_total, _extract_usage(accumulated))
        # Flush any remaining buffered content from the think-tag parser.
        flush_visible, flush_thinking = think_tag_parser.flush()
        if flush_thinking:
            collected_reasoning_parts.append(flush_thinking)
            if supports_reasoning:
                writer("\x00REASONING:" + json.dumps(flush_thinking))
        if flush_visible:
            collected_parts.append(flush_visible)
            writer(flush_visible)
    except Exception as exc:
        error_message = f"Error: {exc}"
        _persist_assistant_message(
            db=db,
            thread_id=state["thread_id"],
            content=error_message,
            model_name=state["selected_model_id"],
            provider_id=state["provider_id"],
            parent_message_id=state.get("parent_message_id"),
            branch_index=state.get("next_branch_index", 0),
        )
        writer(error_message)
    else:
        assistant_content = "".join(collected_parts).strip() or "(No response)"
        ended_at = time.perf_counter()
        latency_ms = int((ended_at - started_at) * 1000)
        ttft_ms = (
            int((first_token_at - started_at) * 1000)
            if first_token_at is not None
            else None
        )
        metrics = {
            "prompt_tokens": (usage_total or {}).get("input_tokens"),
            "completion_tokens": (usage_total or {}).get("output_tokens"),
            "total_tokens": (usage_total or {}).get("total_tokens"),
            "latency_ms": latency_ms,
            "time_to_first_token_ms": ttft_ms,
        }
        assistant_reasoning = "".join(collected_reasoning_parts).strip() or None
        message_id = _persist_assistant_message(
            db=db,
            thread_id=state["thread_id"],
            content=assistant_content,
            model_name=state["selected_model_id"],
            provider_id=state["provider_id"],
            citations=retrieved_chunks,
            parent_message_id=state.get("parent_message_id"),
            branch_index=state.get("next_branch_index", 0),
            metrics=metrics,
            reasoning_content=assistant_reasoning,
        )
        if retrieved_chunks:
            writer("\x00CITATIONS:" + json.dumps(list(retrieved_chunks)))
        if message_id:
            writer("\x00MESSAGE_ID:" + message_id)
        writer("\x00METRICS:" + json.dumps(metrics))
        run_id = config.get("run_id")
        if run_id and os.environ.get("LANGSMITH_API_KEY"):
            writer(f"\x00TRACE_URL:https://smith.langchain.com/runs/{run_id}")
        # Update thread summary and user facts after each response (best-effort).
        asyncio.create_task(_run_memory_update(
            db=db,
            thread_id=state["thread_id"],
            user_id=state["user_id"],
            llm=llm,
            dropped_history_message_ids=list(state.get("dropped_history_message_ids") or []),
        ))
    return {}


async def _run_memory_update(
    *,
    db,
    thread_id: str,
    user_id: str,
    llm,
    dropped_history_message_ids: list[str],
) -> None:
    try:
        await update_memory(
            db=db,
            thread_id=thread_id,
            user_id=user_id,
            llm=llm,
            dropped_history_message_ids=dropped_history_message_ids,
        )
    except Exception:
        pass
