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
    supports_reasoning: bool = False


def _infer_reasoning_support(provider: ProviderCode, model_id: str) -> bool:
    lower = model_id.lower()
    if provider == ProviderCode.OPENAI:
        # o-series always reason; gpt-5 / gpt-5-mini / bare point-releases
        # (gpt-5.2) reason too, but versioned suffixed variants (gpt-5.4-mini)
        # are standard chat models that don't surface thinking tokens.
        # o-series surfaces reasoning via Chat Completions.
        # gpt-5 and bare point-releases (gpt-5.2) do too; -mini variants do not.
        return bool(re.match(r"^o\d|^gpt-5$|^gpt-5\.\d+$", lower))
    if provider == ProviderCode.ANTHROPIC:
        return bool(re.match(r"^claude-(opus|sonnet|haiku)-4", lower))
    if provider == ProviderCode.GEMINI:
        return bool(re.match(r"^gemini-(2\.5|3)", lower))
    if provider == ProviderCode.GROQ:
        return bool(re.search(r"deepseek-r1|reasoning", lower))
    return False


def _is_chat_model(
    model_id: str,
    include_prefixes: tuple[str, ...],
    skip_substrings: tuple[str, ...] = (),
    snapshot_pattern: re.Pattern[str] | None = None,
    skip_latest: bool = False,
) -> bool:
    lower = model_id.lower()
    if not any(lower.startswith(p) for p in include_prefixes):
        return False
    if any(s in lower for s in skip_substrings):
        return False
    if skip_latest and lower.endswith("-latest"):
        return False
    if snapshot_pattern and snapshot_pattern.search(model_id):
        return False
    return True


_OPENAI_DATE_SUFFIX = re.compile(r"-20\d{2}-\d{2}")
_OPENAI_INCLUDE_PREFIXES = ("gpt-5",)
_OPENAI_SKIP_SUBSTRINGS = (
    "realtime", "audio", "tts", "whisper", "dall-e",
    "embedding", "moderation", "transcribe", "search",
    "image", "sora", "codex", "chat-latest",
)

_ANTHROPIC_INCLUDE_PREFIXES = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4")

_GEMINI_SNAPSHOT_SUFFIX = re.compile(r"-\d{3}$|-\d{2}-\d{4}$")
_GEMINI_INCLUDE_PREFIXES = ("gemini-2.5", "gemini-3.1", "gemini-3")
_GEMINI_SKIP_SUBSTRINGS = ("tts", "gemini-3-pro", "nano", "customtools", "image", "deep-research")

_GROQ_INCLUDE_PREFIXES = ("openai/gpt-oss", "llama-3.3", "llama-3.1")
_GROQ_SKIP_SUBSTRINGS = ("safeguard",)


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
        if not _is_chat_model(model_id, _OPENAI_INCLUDE_PREFIXES, _OPENAI_SKIP_SUBSTRINGS, _OPENAI_DATE_SUFFIX):
            continue
        display_name = model_id.replace("-", " ").title().replace("Gpt", "GPT").replace("Oss", "OSS")
        models.append(
            _FetchedModel(
                model_id=model_id,
                display_name=display_name,
                supports_reasoning=_infer_reasoning_support(ProviderCode.OPENAI, model_id),
            )
        )
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
        if not _is_chat_model(model_id, _ANTHROPIC_INCLUDE_PREFIXES):
            continue
        display_name: str = item.get("display_name", model_id)
        models.append(
            _FetchedModel(
                model_id=model_id,
                display_name=display_name,
                supports_reasoning=_infer_reasoning_support(ProviderCode.ANTHROPIC, model_id),
            )
        )
    return models


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
        if not _is_chat_model(model_id, _GEMINI_INCLUDE_PREFIXES, _GEMINI_SKIP_SUBSTRINGS, _GEMINI_SNAPSHOT_SUFFIX, skip_latest=True):
            continue
        display_name: str = item.get("displayName", model_id)
        models.append(
            _FetchedModel(
                model_id=model_id,
                display_name=display_name,
                supports_reasoning=_infer_reasoning_support(ProviderCode.GEMINI, model_id),
            )
        )
    return models


def _fetch_groq_models(api_key: str) -> list[_FetchedModel]:
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()

    models: list[_FetchedModel] = []
    for item in resp.json().get("data", []):
        model_id: str = item.get("id", "")
        if not _is_chat_model(model_id, _GROQ_INCLUDE_PREFIXES, _GROQ_SKIP_SUBSTRINGS):
            continue
        display_name = model_id.replace("-", " ").replace("_", " ").title().replace("Gpt", "GPT").replace("Oss", "OSS")
        models.append(
            _FetchedModel(
                model_id=model_id,
                display_name=display_name,
                supports_reasoning=_infer_reasoning_support(ProviderCode.GROQ, model_id),
            )
        )
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
                    supports_reasoning=fm.supports_reasoning,
                )
            )
            stats.models_added += 1
        else:
            changed = (
                stored.display_name != fm.display_name
                or not stored.is_active
                or stored.supports_reasoning != fm.supports_reasoning
            )
            if changed:
                stored.display_name = fm.display_name
                stored.is_active = True
                stored.supports_reasoning = fm.supports_reasoning
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
