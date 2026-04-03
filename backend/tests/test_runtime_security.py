from __future__ import annotations

import pytest

from app import runtime_config, security
from app.services import auth


@pytest.fixture(autouse=True)
def reset_env_state(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in (
        "APP_ENV",
        "ENVIRONMENT",
        "NODE_ENV",
        "AUTH_SECRET",
        "API_KEY_ENCRYPTION_KEY",
        "BACKEND_CORS_ORIGINS",
        "CORS_ALLOWED_ORIGINS",
        "AUTH_GOOGLE_CLIENT_ID",
        "AUTH_GOOGLE_CLIENT_SECRET",
    ):
        monkeypatch.delenv(env_name, raising=False)

    runtime_config.get_cors_allowed_origins.cache_clear()
    security._get_fernet.cache_clear()


def test_encrypt_secret_requires_configured_key() -> None:
    with pytest.raises(security.EncryptionConfigError, match="API_KEY_ENCRYPTION_KEY is not set"):
        security.encrypt_secret("provider-key")


def test_get_auth_secret_requires_configuration() -> None:
    with pytest.raises(auth.AuthConfigError, match="AUTH_SECRET must be configured"):
        auth._get_auth_secret()


def test_list_oauth_providers_disabled_without_auth_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("AUTH_GOOGLE_CLIENT_SECRET", "client-secret")

    providers = auth.list_oauth_providers()

    google_provider = next(provider for provider in providers if provider["code"] == "google")
    assert google_provider["enabled"] is False


def test_get_cors_allowed_origins_reads_configured_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BACKEND_CORS_ORIGINS", "https://app.example.com, https://admin.example.com/")

    assert runtime_config.get_cors_allowed_origins() == [
        "https://app.example.com",
        "https://admin.example.com",
    ]


def test_get_cors_allowed_origins_requires_configuration() -> None:
    with pytest.raises(runtime_config.RuntimeConfigError, match="BACKEND_CORS_ORIGINS must be configured"):
        runtime_config.get_cors_allowed_origins()
