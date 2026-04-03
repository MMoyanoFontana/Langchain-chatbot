from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException, status
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
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
    _resolve_provider_api_key_value,
)


# ---------------------------------------------------------------------------
# LLM client construction
# ---------------------------------------------------------------------------

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
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)
    return ""


def _build_chat_client(
    provider_code: ProviderCode,
    model_name: str,
    api_key: str,
):
    if provider_code == ProviderCode.OPENAI:
        return ChatOpenAI(model=model_name, streaming=True, temperature=0.2, api_key=api_key)

    if provider_code == ProviderCode.GROQ:
        return ChatOpenAI(
            model=model_name,
            streaming=True,
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
        return ChatAnthropic(model=model_name, temperature=0.2, streaming=True, api_key=api_key)

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
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.2, google_api_key=api_key)

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

async def _stream_model_and_persist(
    state: ChatModelStreamState,
    config: RunnableConfig,
) -> dict[str, str]:
    db = _get_db(config)
    writer = get_stream_writer()
    collected_parts: list[str] = []
    retrieved_chunks: list[RetrievedChunkState] = []
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
        )

        messages = [
            SystemMessage(content=ASSISTANT_SYSTEM_PROMPT),
            HumanMessage(content=state["prompt"]),
        ]

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

        try:
            async for chunk in llm_with_tools.astream(messages):
                chunk_text = _extract_chunk_text(chunk.content)
                if chunk_text:
                    collected_parts.append(chunk_text)
                    writer(chunk_text)
                accumulated = chunk if accumulated is None else accumulated + chunk  # type: ignore[operator]
        except (NotImplementedError, AttributeError):
            # Model doesn't support tool calling — fall back to plain streaming,
            # but only if we haven't already emitted any tokens to the client.
            if accumulated is not None or collected_parts:
                raise
            tools_supported = False
            collected_parts = []
            accumulated = None
            async for chunk in llm.astream(messages):
                chunk_text = _extract_chunk_text(chunk.content)
                if chunk_text:
                    collected_parts.append(chunk_text)
                    writer(chunk_text)

        # If the model decided to call a tool, execute it and stream the final answer.
        if tools_supported and accumulated is not None and accumulated.tool_calls:
            tool_call = accumulated.tool_calls[0]
            tool_content, retrieved_chunks = await _execute_tool_call(tool_call, state, config)
            tool_message = ToolMessage(
                content=tool_content,
                tool_call_id=tool_call["id"],
            )
            collected_parts = []
            async for chunk in llm.astream(messages + [accumulated, tool_message]):
                chunk_text = _extract_chunk_text(chunk.content)
                if chunk_text:
                    collected_parts.append(chunk_text)
                    writer(chunk_text)
    except Exception as exc:
        error_message = f"Error: {exc}"
        _persist_assistant_message(
            db=db,
            thread_id=state["thread_id"],
            content=error_message,
            model_name=state["selected_model_id"],
            provider_id=state["provider_id"],
        )
        writer(error_message)
    else:
        assistant_content = "".join(collected_parts).strip() or "(No response)"
        _persist_assistant_message(
            db=db,
            thread_id=state["thread_id"],
            content=assistant_content,
            model_name=state["selected_model_id"],
            provider_id=state["provider_id"],
            citations=retrieved_chunks,
        )
        if retrieved_chunks:
            writer("\x00CITATIONS:" + json.dumps(list(retrieved_chunks)))
        # Update thread summary and user facts after each response (best-effort).
        asyncio.ensure_future(_run_memory_update(
            db=db,
            thread_id=state["thread_id"],
            user_id=state["user_id"],
            llm=llm,
        ))
    return {}


async def _run_memory_update(*, db, thread_id: str, user_id: str, llm) -> None:
    try:
        await update_memory(db=db, thread_id=thread_id, user_id=user_id, llm=llm)
    except Exception:
        pass
