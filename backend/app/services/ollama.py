"""Helpers for local OpenAI-compatible providers (Ollama, LM Studio)."""
from __future__ import annotations

import httpx


class LocalModelServerError(RuntimeError):
    """Raised when the local model server is unreachable or returns bad data."""


def normalize_local_base_url(raw_url: str) -> str:
    """Return the stored endpoint URL with the OpenAI-compatible ``/v1`` path.

    Local servers expose their OpenAI-compatible API under ``/v1``
    (e.g. ``http://localhost:11434/v1``); append it when absent so users can
    store the bare host URL.
    """
    base = raw_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


async def fetch_ollama_models(base_url: str) -> list[dict[str, str]]:
    """Fetch available model ids from a local OpenAI-compatible server.

    Args:
        base_url: The stored endpoint URL (with or without ``/v1``).

    Returns:
        A list of ``{"model_id": ..., "display_name": ...}`` dicts.

    Raises:
        LocalModelServerError: If the server is unreachable, returns an error
            status, or responds with invalid JSON.
    """
    url = f"{normalize_local_base_url(base_url)}/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise LocalModelServerError(
            f"Could not reach local model server at {base_url}: {exc}"
        ) from exc
    except ValueError as exc:
        raise LocalModelServerError(
            f"Invalid JSON response from local model server at {base_url}."
        ) from exc

    models = data.get("data") if isinstance(data, dict) else None
    return [
        {"model_id": m["id"], "display_name": m["id"]}
        for m in models or []
        if isinstance(m, dict) and m.get("id")
    ]
