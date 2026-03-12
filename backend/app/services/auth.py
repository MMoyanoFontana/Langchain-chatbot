from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.models import AuthIdentity, AuthProvider, AuthSession, User, utc_now

PASSWORD_HASH_NAME = "scrypt"
PASSWORD_HASH_N = 2**17
PASSWORD_HASH_R = 8
PASSWORD_HASH_P = 1
PASSWORD_HASH_DKLEN = 64
SESSION_MAX_AGE = timedelta(days=int(os.getenv("AUTH_SESSION_TTL_DAYS", "30")))
STATE_MAX_AGE = timedelta(minutes=int(os.getenv("AUTH_STATE_TTL_MINUTES", "10")))
PLACEHOLDER_EMAIL_DOMAIN = "users.local"
SESSION_TOKEN_BYTES = 48
PRODUCTION_ENV_VALUES = {"prod", "production"}
LOGGER = logging.getLogger(__name__)
_DEV_FALLBACK_AUTH_SECRET: bytes | None = None


class AuthConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class OAuthProviderDefinition:
    code: AuthProvider
    label: str
    authorize_url: str
    token_url: str
    client_id_env: str
    client_secret_env: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class OAuthProviderConfig:
    definition: OAuthProviderDefinition
    client_id: str
    client_secret: str


@dataclass(frozen=True)
class OAuthUserInfo:
    subject: str
    email: str | None
    full_name: str | None
    avatar_url: str | None


OAUTH_PROVIDERS: dict[AuthProvider, OAuthProviderDefinition] = {
    AuthProvider.GOOGLE: OAuthProviderDefinition(
        code=AuthProvider.GOOGLE,
        label="Google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        client_id_env="AUTH_GOOGLE_CLIENT_ID",
        client_secret_env="AUTH_GOOGLE_CLIENT_SECRET",
        scopes=("openid", "email", "profile"),
    ),
    AuthProvider.GITHUB: OAuthProviderDefinition(
        code=AuthProvider.GITHUB,
        label="GitHub",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        client_id_env="AUTH_GITHUB_CLIENT_ID",
        client_secret_env="AUTH_GITHUB_CLIENT_SECRET",
        scopes=("read:user", "user:email"),
    ),
    AuthProvider.MICROSOFT: OAuthProviderDefinition(
        code=AuthProvider.MICROSOFT,
        label="Microsoft",
        authorize_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        client_id_env="AUTH_MICROSOFT_CLIENT_ID",
        client_secret_env="AUTH_MICROSOFT_CLIENT_SECRET",
        scopes=("openid", "email", "profile"),
    ),
}


def _normalize_email(email: str | None) -> str | None:
    if email is None:
        return None
    normalized = email.strip().lower()
    return normalized or None


def normalize_return_to(return_to: str | None) -> str:
    if not return_to:
        return "/"
    normalized = return_to.strip()
    if not normalized.startswith("/") or normalized.startswith("//"):
        return "/"
    return normalized or "/"


def _normalize_name(full_name: str | None) -> str | None:
    if full_name is None:
        return None
    normalized = full_name.strip()
    return normalized or None


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _base64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(f"{raw}{padding}".encode("utf-8"))


def _hash_password_value(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=PASSWORD_HASH_N,
        r=PASSWORD_HASH_R,
        p=PASSWORD_HASH_P,
        dklen=PASSWORD_HASH_DKLEN,
    )


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    salt = os.urandom(16)
    derived_key = _hash_password_value(password, salt)
    return (
        f"{PASSWORD_HASH_NAME}${PASSWORD_HASH_N}${PASSWORD_HASH_R}${PASSWORD_HASH_P}$"
        f"{_base64url_encode(salt)}${_base64url_encode(derived_key)}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, raw_n, raw_r, raw_p, encoded_salt, encoded_hash = password_hash.split("$", 5)
    except ValueError:
        return False

    if algorithm != PASSWORD_HASH_NAME:
        return False

    try:
        n = int(raw_n)
        r = int(raw_r)
        p = int(raw_p)
        salt = _base64url_decode(encoded_salt)
        expected_hash = _base64url_decode(encoded_hash)
    except (TypeError, ValueError):
        return False

    derived_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected_hash),
    )
    return hmac.compare_digest(derived_hash, expected_hash)


def _get_auth_secret() -> bytes:
    configured_secret = os.getenv("AUTH_SECRET", "").strip()
    if configured_secret:
        return configured_secret.encode("utf-8")

    environment_values = (
        os.getenv("APP_ENV", "").strip().lower(),
        os.getenv("ENVIRONMENT", "").strip().lower(),
        os.getenv("NODE_ENV", "").strip().lower(),
    )
    if any(value in PRODUCTION_ENV_VALUES for value in environment_values if value):
        raise AuthConfigError("AUTH_SECRET must be configured in production.")

    global _DEV_FALLBACK_AUTH_SECRET
    if _DEV_FALLBACK_AUTH_SECRET is None:
        _DEV_FALLBACK_AUTH_SECRET = os.urandom(32)
        LOGGER.warning(
            "AUTH_SECRET is not set; using an ephemeral development secret. "
            "OAuth state signatures will be invalidated on restart."
        )
    return _DEV_FALLBACK_AUTH_SECRET


def _build_session_token() -> str:
    return _base64url_encode(os.urandom(SESSION_TOKEN_BYTES))


def _hash_session_token(session_token: str) -> str:
    return hashlib.sha256(session_token.encode("utf-8")).hexdigest()


def _is_session_expired(expires_at: datetime) -> bool:
    normalized_expiry = (
        expires_at.replace(tzinfo=timezone.utc)
        if expires_at.tzinfo is None
        else expires_at
    )
    return normalized_expiry <= utc_now()


def create_user_session(db: Session, user: User) -> str:
    session_token = _build_session_token()
    db.add(
        AuthSession(
            user_id=user.id,
            token_hash=_hash_session_token(session_token),
            expires_at=utc_now() + SESSION_MAX_AGE,
        )
    )
    db.commit()
    return session_token


def revoke_session_token(db: Session, session_token: str | None) -> None:
    if not session_token:
        return
    session = db.scalar(select(AuthSession).where(AuthSession.token_hash == _hash_session_token(session_token)))
    if session is None:
        return
    db.delete(session)
    db.commit()
    return


def _get_session_record(db: Session, session_token: str) -> AuthSession | None:
    if not session_token:
        return None
    return db.scalar(
        select(AuthSession)
        .where(AuthSession.token_hash == _hash_session_token(session_token))
        .options(joinedload(AuthSession.user))
    )


def get_user_for_session_token(db: Session, session_token: str | None) -> User | None:
    if not session_token:
        return None

    auth_session = _get_session_record(db, session_token)
    if auth_session is None:
        return None

    if _is_session_expired(auth_session.expires_at):
        db.delete(auth_session)
        db.commit()
        return None

    user = auth_session.user
    if user is None or not user.is_active:
        return None

    return user


def _placeholder_email(provider: AuthProvider, subject: str) -> str:
    normalized_subject = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-") or "user"
    return f"{provider.value}-{normalized_subject[:48]}@{PLACEHOLDER_EMAIL_DOMAIN}"


def register_user_with_password(
    db: Session,
    *,
    email: str,
    password: str,
    full_name: str | None,
) -> tuple[User, str]:
    normalized_email = _normalize_email(email)
    if not normalized_email:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email cannot be blank.")

    user = db.scalar(select(User).where(User.email == normalized_email))
    if user is not None and user.password_hash is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    if user is not None and not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is disabled.",
        )

    password_hash = hash_password(password)
    normalized_name = _normalize_name(full_name)
    if user is None:
        user = User(email=normalized_email, full_name=normalized_name, password_hash=password_hash)
        db.add(user)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            ) from exc
        db.refresh(user)
    else:
        user.password_hash = password_hash
        if normalized_name:
            user.full_name = normalized_name
        db.commit()
        db.refresh(user)

    return user, create_user_session(db, user)


def login_user_with_password(db: Session, *, email: str, password: str) -> tuple[User, str]:
    normalized_email = _normalize_email(email)
    if not normalized_email:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email cannot be blank.")

    user = db.scalar(select(User).where(User.email == normalized_email))
    if user is None or user.password_hash is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is disabled.",
        )

    return user, create_user_session(db, user)


def _build_state_signature(encoded_payload: str) -> str:
    try:
        secret = _get_auth_secret()
    except AuthConfigError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    digest = hmac.new(secret, encoded_payload.encode("utf-8"), hashlib.sha256).digest()
    return _base64url_encode(digest)


def _encode_state_payload(payload: dict[str, Any]) -> str:
    raw_payload = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _base64url_encode(raw_payload)


def _decode_state_payload(encoded_payload: str) -> dict[str, Any]:
    raw_payload = _base64url_decode(encoded_payload)
    payload = json.loads(raw_payload.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("State payload must be an object.")
    return payload


def _sign_oauth_state(payload: dict[str, Any]) -> str:
    encoded_payload = _encode_state_payload(payload)
    return f"{encoded_payload}.{_build_state_signature(encoded_payload)}"


def _verify_oauth_state(state: str) -> dict[str, Any]:
    encoded_payload, separator, encoded_signature = state.partition(".")
    if not encoded_payload or not separator or not encoded_signature:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth state is invalid.")

    expected_signature = _build_state_signature(encoded_payload)
    if not hmac.compare_digest(encoded_signature, expected_signature):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth state is invalid.")

    try:
        payload = _decode_state_payload(encoded_payload)
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth state is invalid.") from None

    expires_at = payload.get("exp")
    if not isinstance(expires_at, int):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth state is invalid.")
    if expires_at < int(utc_now().timestamp()):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth sign-in request has expired.")

    return payload


def _get_oauth_provider_config(provider: AuthProvider) -> OAuthProviderConfig | None:
    definition = OAUTH_PROVIDERS[provider]
    client_id = os.getenv(definition.client_id_env, "").strip()
    client_secret = os.getenv(definition.client_secret_env, "").strip()
    if not client_id or not client_secret:
        return None
    return OAuthProviderConfig(definition=definition, client_id=client_id, client_secret=client_secret)


def list_oauth_providers() -> list[dict[str, str | bool]]:
    providers: list[dict[str, str | bool]] = []
    for provider in OAUTH_PROVIDERS.values():
        providers.append(
            {
                "code": provider.code.value,
                "label": provider.label,
                "enabled": _get_oauth_provider_config(provider.code) is not None,
            }
        )
    return providers


def build_oauth_authorize_url(
    provider: AuthProvider,
    *,
    redirect_uri: str,
    return_to: str | None,
) -> str:
    oauth_config = _get_oauth_provider_config(provider)
    if oauth_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{OAUTH_PROVIDERS[provider].label} sign-in is not configured.",
        )

    normalized_return_to = normalize_return_to(return_to)
    state = _sign_oauth_state(
        {
            "provider": provider.value,
            "redirect_uri": redirect_uri,
            "return_to": normalized_return_to,
            "exp": int((utc_now() + STATE_MAX_AGE).timestamp()),
        }
    )

    params = {
        "client_id": oauth_config.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(oauth_config.definition.scopes),
        "state": state,
    }
    if provider == AuthProvider.GOOGLE:
        params["access_type"] = "offline"
        params["prompt"] = "select_account"

    return f"{oauth_config.definition.authorize_url}?{urlencode(params)}"


async def exchange_oauth_code(
    db: Session,
    *,
    provider: AuthProvider,
    code: str,
    state: str,
    redirect_uri: str,
) -> tuple[User, str, str]:
    oauth_config = _get_oauth_provider_config(provider)
    if oauth_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{OAUTH_PROVIDERS[provider].label} sign-in is not configured.",
        )

    state_payload = _verify_oauth_state(state)
    if state_payload.get("provider") != provider.value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth provider does not match request.")
    if state_payload.get("redirect_uri") != redirect_uri:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth redirect URI does not match request.")

    async with httpx.AsyncClient(timeout=10.0) as client:
        token_payload = await _request_oauth_token(
            client,
            oauth_config=oauth_config,
            code=code,
            redirect_uri=redirect_uri,
        )
        user_info = await _request_oauth_user_info(client, provider, token_payload)

    user = _upsert_oauth_user(
        db,
        provider=provider,
        user_info=user_info,
    )
    return user, create_user_session(db, user), normalize_return_to(state_payload.get("return_to"))


async def _request_oauth_token(
    client: httpx.AsyncClient,
    *,
    oauth_config: OAuthProviderConfig,
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    payload = {
        "client_id": oauth_config.client_id,
        "client_secret": oauth_config.client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    response = await client.post(
        oauth_config.definition.token_url,
        headers=headers,
        data=payload,
    )

    data = await _parse_oauth_response(response, fallback="OAuth token exchange failed.")
    access_token = data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="OAuth token exchange did not return an access token.")
    return data


async def _parse_oauth_response(response: httpx.Response, *, fallback: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {}

    if not response.is_success:
        detail = payload.get("error_description") or payload.get("error") or fallback
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(detail))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=fallback)
    return payload


async def _request_oauth_user_info(
    client: httpx.AsyncClient,
    provider: AuthProvider,
    token_payload: dict[str, Any],
) -> OAuthUserInfo:
    access_token = str(token_payload["access_token"])
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    if provider == AuthProvider.GOOGLE:
        response = await client.get("https://openidconnect.googleapis.com/v1/userinfo", headers=headers)
        payload = await _parse_oauth_response(response, fallback="Google profile request failed.")
        subject = str(payload.get("sub") or "").strip()
        return _build_oauth_user_info(
            subject=subject,
            email=payload.get("email"),
            full_name=payload.get("name"),
            avatar_url=payload.get("picture"),
        )

    if provider == AuthProvider.GITHUB:
        user_headers = {
            **headers,
            "User-Agent": "Langchain-Chatbot",
        }
        profile_response = await client.get("https://api.github.com/user", headers=user_headers)
        profile_payload = await _parse_oauth_response(profile_response, fallback="GitHub profile request failed.")
        emails_response = await client.get("https://api.github.com/user/emails", headers=user_headers)
        emails_payload = await _parse_oauth_response(emails_response, fallback="GitHub email request failed.")
        email = _select_github_email(emails_payload)
        subject = str(profile_payload.get("id") or "").strip()
        return _build_oauth_user_info(
            subject=subject,
            email=email,
            full_name=profile_payload.get("name") or profile_payload.get("login"),
            avatar_url=profile_payload.get("avatar_url"),
        )

    if provider == AuthProvider.MICROSOFT:
        response = await client.get("https://graph.microsoft.com/oidc/userinfo", headers=headers)
        payload = await _parse_oauth_response(response, fallback="Microsoft profile request failed.")
        subject = str(payload.get("sub") or "").strip()
        return _build_oauth_user_info(
            subject=subject,
            email=payload.get("email") or payload.get("preferred_username"),
            full_name=payload.get("name"),
            avatar_url=None,
        )

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported OAuth provider.")


def _select_github_email(payload: Any) -> str | None:
    if not isinstance(payload, list):
        return None

    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("primary") and item.get("verified") and isinstance(item.get("email"), str):
            return item["email"]

    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("verified") and isinstance(item.get("email"), str):
            return item["email"]

    for item in payload:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("email"), str):
            return item["email"]

    return None


def _build_oauth_user_info(
    *,
    subject: str,
    email: Any,
    full_name: Any,
    avatar_url: Any,
) -> OAuthUserInfo:
    normalized_subject = subject.strip()
    if not normalized_subject:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="OAuth provider did not return a valid account identifier.")

    return OAuthUserInfo(
        subject=normalized_subject,
        email=_normalize_email(email if isinstance(email, str) else None),
        full_name=_normalize_name(full_name if isinstance(full_name, str) else None),
        avatar_url=avatar_url.strip() if isinstance(avatar_url, str) and avatar_url.strip() else None,
    )


def _upsert_oauth_user(
    db: Session,
    *,
    provider: AuthProvider,
    user_info: OAuthUserInfo,
) -> User:
    identity = db.scalar(
        select(AuthIdentity)
        .where(
            AuthIdentity.provider == provider,
            AuthIdentity.provider_subject == user_info.subject,
        )
        .options(joinedload(AuthIdentity.user))
    )

    if identity is not None:
        if identity.user is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OAuth identity is missing a user.")

        user = identity.user
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account is disabled.")
        _merge_user_profile(db, user, user_info)
        identity.email = user_info.email
        identity.avatar_url = user_info.avatar_url
        db.commit()
        db.refresh(user)
        return user

    user_email = user_info.email or _placeholder_email(provider, user_info.subject)
    user = db.scalar(select(User).where(User.email == user_email))
    if user is None:
        user = User(
            email=user_email,
            full_name=user_info.full_name,
            avatar_url=user_info.avatar_url,
        )
        db.add(user)
        db.flush()
    else:
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account is disabled.")
        _merge_user_profile(db, user, user_info)

    identity = AuthIdentity(
        user_id=user.id,
        provider=provider,
        provider_subject=user_info.subject,
        email=user_info.email,
        avatar_url=user_info.avatar_url,
    )
    db.add(identity)
    db.commit()
    db.refresh(user)
    return user


def _merge_user_profile(db: Session, user: User, user_info: OAuthUserInfo) -> None:
    if user_info.email:
        existing_owner = db.scalar(select(User).where(User.email == user_info.email))
        if existing_owner is None or existing_owner.id == user.id:
            user.email = user_info.email
    if user_info.full_name:
        user.full_name = user_info.full_name
    if user_info.avatar_url:
        user.avatar_url = user_info.avatar_url
