from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Provider, ProviderCode, ProviderModel


@dataclass(frozen=True)
class CatalogModelDefinition:
    model_id: str
    display_name: str


@dataclass(frozen=True)
class CatalogProviderDefinition:
    code: ProviderCode
    display_name: str
    models: tuple[CatalogModelDefinition, ...]
    deactivate_unlisted_models: bool = True


RECOMMENDED_PROVIDER_CATALOG: tuple[CatalogProviderDefinition, ...] = (
    CatalogProviderDefinition(
        code=ProviderCode.OPENAI,
        display_name="OpenAI",
        models=(
            CatalogModelDefinition(model_id="gpt-5.2", display_name="GPT-5.2"),
            CatalogModelDefinition(model_id="gpt-5", display_name="GPT-5"),
            CatalogModelDefinition(model_id="gpt-5-mini", display_name="GPT-5 mini"),
        ),
    ),
    CatalogProviderDefinition(
        code=ProviderCode.ANTHROPIC,
        display_name="Anthropic",
        models=(
            CatalogModelDefinition(
                model_id="claude-opus-4-1-20250805",
                display_name="Claude Opus 4.1",
            ),
            CatalogModelDefinition(
                model_id="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
            ),
        ),
    ),
    CatalogProviderDefinition(
        code=ProviderCode.GEMINI,
        display_name="Google Gemini",
        models=(
            CatalogModelDefinition(model_id="gemini-2.5-pro", display_name="Gemini 2.5 Pro"),
            CatalogModelDefinition(model_id="gemini-2.5-flash", display_name="Gemini 2.5 Flash"),
        ),
    ),
    CatalogProviderDefinition(
        code=ProviderCode.GROQ,
        display_name="Groq",
        models=(
            CatalogModelDefinition(model_id="openai/gpt-oss-120b", display_name="GPT-OSS 120B"),
            CatalogModelDefinition(model_id="openai/gpt-oss-20b", display_name="GPT-OSS 20B"),
            CatalogModelDefinition(
                model_id="meta-llama/llama-4-scout-17b-16e-instruct",
                display_name="Llama 4 Scout 17B",
            ),
        ),
    ),
    CatalogProviderDefinition(
        code=ProviderCode.XAI,
        display_name="xAI",
        models=(
            CatalogModelDefinition(model_id="grok-3-latest", display_name="Grok 3 Latest"),
            CatalogModelDefinition(
                model_id="grok-3-fast-latest",
                display_name="Grok 3 Fast Latest",
            ),
            CatalogModelDefinition(model_id="grok-code-fast-1", display_name="Grok Code Fast 1"),
        ),
    ),
    CatalogProviderDefinition(
        code=ProviderCode.OPENROUTER,
        display_name="OpenRouter",
        models=(
            CatalogModelDefinition(
                model_id="anthropic/claude-sonnet-4.5",
                display_name="Claude Sonnet 4.5 (OpenRouter)",
            ),
            CatalogModelDefinition(
                model_id="openai/gpt-5.1",
                display_name="GPT-5.1 (OpenRouter)",
            ),
            CatalogModelDefinition(
                model_id="google/gemini-3-pro-preview",
                display_name="Gemini 3 Pro Preview (OpenRouter)",
            ),
        ),
    ),
    CatalogProviderDefinition(
        code=ProviderCode.OTHER,
        display_name="Other",
        models=(),
        deactivate_unlisted_models=False,
    ),
)


def seed_provider_catalog(db: Session) -> None:
    existing_providers = {
        provider.code: provider for provider in db.scalars(select(Provider)).all()
    }

    for provider_definition in RECOMMENDED_PROVIDER_CATALOG:
        provider = existing_providers.get(provider_definition.code)
        if provider is None:
            provider = Provider(
                code=provider_definition.code,
                display_name=provider_definition.display_name,
                is_active=True,
            )
            db.add(provider)
            db.flush()
            existing_providers[provider.code] = provider
        else:
            provider.display_name = provider_definition.display_name
            provider.is_active = True

        models_for_provider = list(
            db.scalars(select(ProviderModel).where(ProviderModel.provider_id == provider.id)).all()
        )
        models_by_id = {
            model.model_id: model for model in models_for_provider
        }

        recommended_model_ids = {
            model_definition.model_id for model_definition in provider_definition.models
        }
        for model_definition in provider_definition.models:
            stored_model = models_by_id.get(model_definition.model_id)
            if stored_model is None:
                db.add(
                    ProviderModel(
                        provider_id=provider.id,
                        model_id=model_definition.model_id,
                        display_name=model_definition.display_name,
                        is_active=True,
                    )
                )
                continue

            stored_model.display_name = model_definition.display_name
            stored_model.is_active = True

        if provider_definition.deactivate_unlisted_models:
            for stored_model in models_for_provider:
                if stored_model.model_id not in recommended_model_ids:
                    stored_model.is_active = False

    db.commit()
