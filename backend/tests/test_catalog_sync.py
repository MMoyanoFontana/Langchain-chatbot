from __future__ import annotations

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import Provider, ProviderCode, ProviderModel
from app.services import catalog_sync


def _session_factory():
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def _seed_groq_provider(db: Session) -> Provider:
    provider = Provider(code=ProviderCode.GROQ, display_name="Groq", is_active=True)
    db.add(provider)
    db.flush()
    db.add(
        ProviderModel(
            provider_id=provider.id,
            model_id="llama-3.1-8b-instant",
            display_name="Llama 3.1 8B Instant",
            is_active=True,
        )
    )
    db.commit()
    db.refresh(provider)
    return provider


def test_sync_catalog_skips_unauthorized_groq(monkeypatch):
    factory = _session_factory()
    db = factory()
    provider = _seed_groq_provider(db)

    monkeypatch.setenv("GROQ_API_KEY", "bad-key")

    request = httpx.Request("GET", "https://api.groq.com/openai/v1/models")
    response = httpx.Response(401, request=request)

    def _raise_unauthorized(_api_key: str):
        raise httpx.HTTPStatusError("Unauthorized", request=request, response=response)

    monkeypatch.setattr(
        catalog_sync,
        "_SYNC_REGISTRY",
        (catalog_sync._ProviderSyncConfig(ProviderCode.GROQ, "GROQ_API_KEY", _raise_unauthorized),),
    )

    results = catalog_sync.sync_catalog(db)

    assert len(results) == 1
    assert results[0].provider == ProviderCode.GROQ.value
    assert results[0].error == "unauthorized (401)"

    stored = db.get(Provider, provider.id)
    assert stored is not None
    assert stored.models[0].is_active is True
