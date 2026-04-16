"""Fetch live model lists from provider APIs and upsert into the DB."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Provider, ProviderCode, ProviderModel

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-provider fetch helpers
# ---------------------------------------------------------------------------

@dataclass
class _FetchedModel:
    model_id: str
    display_name: str


# Matches dated snapshot suffixes like -2024-05-13 or -2025-08
_OPENAI_DATE_SUFFIX = re.compile(r"-20\d{2}-\d{2}")

# Substrings that disqualify any model (non-chat capabilities)
_OPENAI_SKIP_SUBSTRINGS = (
    "realtime", "audio", "tts", "whisper", "dall-e",
    "embedding", "moderation", "transcribe", "search",
    "image", "sora", "codex", "chat-latest",
)
# Exact IDs to exclude (legacy base model names)
_OPENAI_SKIP_EXACT = {"gpt-4", "gpt-4-base"}
# Prefixes for legacy / non-chat families
_OPENAI_SKIP_PREFIXES = (
    "gpt-3",       # all gpt-3.x
    "gpt-4-",      # gpt-4-turbo, gpt-4-0613, etc. (NOT gpt-4o)
    "chatgpt-",    # chatgpt-4o-latest aliases
    "babbage", "davinci", "curie", "ada-", "canary",
)


def _is_openai_chat_model(model_id: str) -> bool:
    if model_id in _OPENAI_SKIP_EXACT:
        return False
    lower = model_id.lower()
    if any(s in lower for s in _OPENAI_SKIP_SUBSTRINGS):
        return False
    if any(lower.startswith(p) for p in _OPENAI_SKIP_PREFIXES):
        return False
    if _OPENAI_DATE_SUFFIX.search(model_id):  # skip pinned dated snapshots
        return False
    return True


def _fetch_openai_models(api_key: str) -> list[_FetchedModel]:
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()

    models: list[_FetchedModel] = []
    for item in resp.json().get("data", []):
        model_id: str = item.get("id", "")
        if not _is_openai_chat_model(model_id):
            continue
        display_name = model_id.replace("-", " ").replace(".", " ").title()
        models.append(_FetchedModel(model_id=model_id, display_name=display_name))
    return models


def _fetch_anthropic_models(api_key: str) -> list[_FetchedModel]:
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        resp.raise_for_status()

    models: list[_FetchedModel] = []
    for item in resp.json().get("data", []):
        model_id: str = item.get("id", "")
        display_name: str = item.get("display_name", model_id)
        models.append(_FetchedModel(model_id=model_id, display_name=display_name))
    return models


# Dated snapshot suffixes for Gemini: -001, -002 or month-year like -10-2025
_GEMINI_SNAPSHOT_SUFFIX = re.compile(r"-\d{3}$|-\d{2}-\d{4}$")

_GEMINI_SKIP_SUBSTRINGS = (
    "tts", "image", "computer-use", "robotics", "lyria",
    "deep-research", "customtools", "clip",
)
_GEMINI_SKIP_PREFIXES = (
    "gemma-",   # open-weights Gemma family, separate from Gemini
    "nano-",    # internal/experimental
)


def _is_gemini_chat_model(model_id: str) -> bool:
    if not model_id.startswith("gemini-"):
        return False
    lower = model_id.lower()
    if any(s in lower for s in _GEMINI_SKIP_SUBSTRINGS):
        return False
    if any(lower.startswith(p) for p in _GEMINI_SKIP_PREFIXES):
        return False
    if lower.endswith("-latest"):
        return False
    if _GEMINI_SNAPSHOT_SUFFIX.search(model_id):  # pinned dated/versioned snapshots
        return False
    return True


def _fetch_gemini_models(api_key: str) -> list[_FetchedModel]:
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
        )
        resp.raise_for_status()

    models: list[_FetchedModel] = []
    for item in resp.json().get("models", []):
        supported = item.get("supportedGenerationMethods", [])
        if "generateContent" not in supported:
            continue
        # name looks like "models/gemini-2.0-flash"
        name: str = item.get("name", "")
        model_id = name.removeprefix("models/")
        if not _is_gemini_chat_model(model_id):
            continue
        display_name: str = item.get("displayName", model_id)
        models.append(_FetchedModel(model_id=model_id, display_name=display_name))
    return models


def _fetch_groq_models(api_key: str) -> list[_FetchedModel]:
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()

    _SKIP_SUBSTRINGS = ("whisper", "guard", "tts", "embedding", "vision")
    models: list[_FetchedModel] = []
    for item in resp.json().get("data", []):
        model_id: str = item.get("id", "")
        lower = model_id.lower()
        if any(s in lower for s in _SKIP_SUBSTRINGS):
            continue
        display_name = model_id.replace("-", " ").replace("_", " ").title()
        models.append(_FetchedModel(model_id=model_id, display_name=display_name))
    return models


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

@dataclass
class _ProviderSyncConfig:
    code: ProviderCode
    env_key: str
    fetcher: object  # callable(api_key) -> list[_FetchedModel]
    deactivate_unlisted: bool = True


_SYNC_REGISTRY: tuple[_ProviderSyncConfig, ...] = (
    _ProviderSyncConfig(ProviderCode.OPENAI, "OPENAI_API_KEY", _fetch_openai_models),
    _ProviderSyncConfig(ProviderCode.ANTHROPIC, "ANTHROPIC_API_KEY", _fetch_anthropic_models),
    _ProviderSyncConfig(ProviderCode.GEMINI, "GEMINI_API_KEY", _fetch_gemini_models),
    _ProviderSyncConfig(ProviderCode.GROQ, "GROQ_API_KEY", _fetch_groq_models),
)

# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------

@dataclass
class ProviderSyncStats:
    provider: str
    models_added: int = 0
    models_updated: int = 0
    models_deactivated: int = 0
    error: str | None = None


def _upsert_provider_models(
    db: Session,
    provider: Provider,
    fetched: list[_FetchedModel],
    deactivate_unlisted: bool,
) -> ProviderSyncStats:
    stats = ProviderSyncStats(provider=provider.code.value)

    existing: list[ProviderModel] = list(
        db.scalars(
            select(ProviderModel).where(ProviderModel.provider_id == provider.id)
        ).all()
    )
    by_id = {m.model_id: m for m in existing}
    fetched_ids = {m.model_id for m in fetched}

    for fm in fetched:
        stored = by_id.get(fm.model_id)
        if stored is None:
            db.add(
                ProviderModel(
                    provider_id=provider.id,
                    model_id=fm.model_id,
                    display_name=fm.display_name,
                    is_active=True,
                )
            )
            stats.models_added += 1
        else:
            if stored.display_name != fm.display_name or not stored.is_active:
                stored.display_name = fm.display_name
                stored.is_active = True
                stats.models_updated += 1

    if deactivate_unlisted:
        for stored in existing:
            if stored.model_id not in fetched_ids and stored.is_active:
                stored.is_active = False
                stats.models_deactivated += 1

    return stats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sync_catalog(db: Session) -> list[ProviderSyncStats]:
    results: list[ProviderSyncStats] = []

    providers_by_code: dict[ProviderCode, Provider] = {
        p.code: p
        for p in db.scalars(select(Provider)).all()
    }

    for cfg in _SYNC_REGISTRY:
        api_key = os.getenv(cfg.env_key, "").strip()
        if not api_key:
            LOGGER.info("catalog_sync_skip provider=%s reason=no_api_key", cfg.code.value)
            continue

        provider = providers_by_code.get(cfg.code)
        if provider is None:
            LOGGER.warning("catalog_sync_skip provider=%s reason=not_in_db", cfg.code.value)
            continue

        try:
            fetched = cfg.fetcher(api_key)
            stats = _upsert_provider_models(db, provider, fetched, cfg.deactivate_unlisted)
            results.append(stats)
            LOGGER.info(
                "catalog_sync_ok provider=%s added=%d updated=%d deactivated=%d",
                cfg.code.value, stats.models_added, stats.models_updated, stats.models_deactivated,
            )
        except Exception as exc:
            LOGGER.exception("catalog_sync_error provider=%s", cfg.code.value)
            results.append(ProviderSyncStats(provider=cfg.code.value, error=str(exc)))

    db.commit()
    return results
