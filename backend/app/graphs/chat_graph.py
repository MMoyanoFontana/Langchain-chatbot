from __future__ import annotations

from uuid import uuid4

from fastapi import HTTPException, status
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.models import ProviderCode
from app.schemas import ChatRequest
from app.services.rag import get_rag_service

from app.graphs._nodes import (
    CONFIG_DB_KEY,
    CONFIG_RAG_SERVICE_KEY,
    ChatGraphResult,
    ChatGraphState,
    ChatModelStreamState,
    HistoryMessageState,
    RetrievedChunkState,
    _error_node,
    _ingest_attachments,
    _load_history,
    _load_memory,
    _load_thread_history,
    _normalize_optional_text,
    _persist_user_message,
    _require_int,
    _require_str,
    _retrieve_context,
    _route_after_step,
    _serialize_attachments,
    _validate_request,
    _resolve_user_provider_key,
)
from app.graphs._stream import _stream_model_and_persist


def _build_chat_graph():
    graph = StateGraph(ChatGraphState)
    graph.add_node("validate_request", _validate_request)
    graph.add_node("load_thread_history", _load_thread_history)
    graph.add_node("load_memory", _load_memory)
    graph.add_node("resolve_user_provider_key", _resolve_user_provider_key)
    graph.add_node("persist_user_message", _persist_user_message)
    graph.add_node("ingest_attachments", _ingest_attachments)
    graph.add_node("load_history", _load_history)
    graph.add_node("retrieve_context", _retrieve_context)
    graph.add_node("error", _error_node)

    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges(
        "validate_request",
        _route_after_step,
        {"continue": "load_thread_history", "error": "error"},
    )
    graph.add_conditional_edges(
        "load_thread_history",
        _route_after_step,
        {"continue": "load_memory", "error": "error"},
    )
    graph.add_edge("load_memory", "resolve_user_provider_key")
    graph.add_conditional_edges(
        "resolve_user_provider_key",
        _route_after_step,
        {"continue": "persist_user_message", "error": "error"},
    )
    graph.add_conditional_edges(
        "persist_user_message",
        _route_after_step,
        {"continue": "ingest_attachments", "error": "error"},
    )
    graph.add_edge("ingest_attachments", "load_history")
    graph.add_edge("load_history", "retrieve_context")
    graph.add_edge("retrieve_context", END)
    graph.add_edge("error", END)

    return graph.compile()


def _build_model_stream_graph():
    graph = StateGraph(ChatModelStreamState)
    graph.add_node("call_model", _stream_model_and_persist)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()


CHAT_GRAPH = _build_chat_graph()
CHAT_MODEL_STREAM_GRAPH = _build_model_stream_graph()


async def run_chat_graph(
    payload: ChatRequest,
    db: Session,
    *,
    user_id: str,
    rag_service=None,
) -> ChatGraphResult:
    request_thread_id = _normalize_optional_text(payload.thread_id)
    config_thread_id = request_thread_id or f"new-thread-{uuid4()}"

    graph_input: ChatGraphState = {
        "prompt": payload.prompt,
        "attachments": _serialize_attachments(payload.attachments),
        "request_thread_id": request_thread_id,
        "model_id": payload.model_id,
        "provider_code": payload.provider_code.value if payload.provider_code is not None else None,
        "user_id": user_id,
        "regenerate_from_message_id": payload.regenerate_from_message_id,
        "continue_from_message_id": payload.continue_from_message_id,
        "compare_with_user_message_id": payload.compare_with_user_message_id,
    }
    if "system_prompt" in payload.model_fields_set:
        graph_input["request_system_prompt"] = payload.system_prompt
        graph_input["should_update_system_prompt"] = True
    if "title" in payload.model_fields_set and payload.title:
        graph_input["request_title"] = payload.title.strip() or None
    final_state = await CHAT_GRAPH.ainvoke(
        graph_input,
        config={
            "configurable": {
                "thread_id": config_thread_id,
                CONFIG_DB_KEY: db,
                CONFIG_RAG_SERVICE_KEY: rag_service if rag_service is not None else get_rag_service(),
            }
        },
    )

    error_message = final_state.get("error_message")
    if error_message:
        error_status = final_state.get("error_status")
        if not isinstance(error_status, int) or error_status <= 0:
            error_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(status_code=error_status, detail=error_message)

    selected_provider_code = ProviderCode(_require_str(final_state, "selected_provider_code"))
    return ChatGraphResult(
        prompt=_require_str(final_state, "model_prompt"),
        system_addendum=final_state.get("system_addendum") or "",
        history_messages=tuple(final_state.get("history_messages") or []),
        dropped_history_count=int(final_state.get("dropped_history_count") or 0),
        dropped_history_message_ids=tuple(final_state.get("dropped_history_message_ids") or []),
        retrieved_chunks=tuple(final_state.get("retrieved_chunks") or []),
        user_id=_require_str(final_state, "user_id"),
        thread_id=_require_str(final_state, "thread_id"),
        provider_id=_require_int(final_state, "provider_id"),
        provider_api_key_id=_require_str(final_state, "provider_api_key_id"),
        provider_code=selected_provider_code,
        model_id=_require_str(final_state, "selected_model_id"),
        thread_has_documents=bool(final_state.get("thread_has_documents", False)),
        pending_document_ids=tuple(final_state.get("pending_document_ids") or []),
        parent_message_id=final_state.get("parent_message_id"),
        next_branch_index=final_state.get("next_branch_index", 0),
        user_message_id=final_state.get("user_message_id"),
        thread_system_prompt=final_state.get("thread_system_prompt"),
    )


def build_model_stream_input(result: ChatGraphResult) -> ChatModelStreamState:
    return {
        "prompt": result.prompt,
        "system_addendum": result.system_addendum,
        "history_messages": list(result.history_messages),
        "dropped_history_count": result.dropped_history_count,
        "dropped_history_message_ids": list(result.dropped_history_message_ids),
        "retrieved_chunks": list(result.retrieved_chunks),
        "user_id": result.user_id,
        "thread_id": result.thread_id,
        "provider_id": result.provider_id,
        "provider_api_key_id": result.provider_api_key_id,
        "selected_provider_code": result.provider_code.value,
        "selected_model_id": result.model_id,
        "thread_has_documents": result.thread_has_documents,
        "parent_message_id": result.parent_message_id,
        "next_branch_index": result.next_branch_index,
        "thread_system_prompt": result.thread_system_prompt,
    }
