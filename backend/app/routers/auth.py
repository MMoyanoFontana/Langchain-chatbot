from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AuthProvider, User
from app.schemas import (
    AuthProviderRead,
    AuthSessionRead,
    EmailLoginRequest,
    EmailRegisterRequest,
    OAuthAuthorizeRead,
    OAuthExchangeRequest,
    UserRead,
)
from app.services.auth import (
    build_oauth_authorize_url,
    exchange_oauth_code,
    list_oauth_providers,
    login_user_with_password,
    register_user_with_password,
    revoke_session_token,
)
from app.services.current_user import get_session_token_from_request, require_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


def _build_auth_response(
    *,
    user: User,
    session_token: str,
    redirect_to: str | None = None,
) -> AuthSessionRead:
    return AuthSessionRead(
        session_token=session_token,
        user=UserRead.model_validate(user),
        redirect_to=redirect_to,
    )


@router.get("/providers", response_model=list[AuthProviderRead])
def get_auth_providers() -> list[dict[str, str | bool]]:
    return list_oauth_providers()


@router.post("/email/register", response_model=AuthSessionRead, status_code=status.HTTP_201_CREATED)
def register_with_email(payload: EmailRegisterRequest, db: Session = Depends(get_db)) -> AuthSessionRead:
    user, session_token = register_user_with_password(
        db,
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
    )
    return _build_auth_response(user=user, session_token=session_token)


@router.post("/email/login", response_model=AuthSessionRead)
def login_with_email(payload: EmailLoginRequest, db: Session = Depends(get_db)) -> AuthSessionRead:
    user, session_token = login_user_with_password(
        db,
        email=payload.email,
        password=payload.password,
    )
    return _build_auth_response(user=user, session_token=session_token)


@router.get("/session", response_model=UserRead)
def get_current_session(user: User = Depends(require_current_user)) -> User:
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    revoke_session_token(db, get_session_token_from_request(request))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


_OAUTH_NONCE_COOKIE = "oauth_nonce"
_OAUTH_NONCE_MAX_AGE = 600  # 10 minutes, matches STATE_MAX_AGE


@router.get("/oauth/{provider}/authorize", response_model=OAuthAuthorizeRead)
def get_oauth_authorize_url(
    provider: AuthProvider,
    response: Response,
    redirect_uri: str = Query(min_length=1, max_length=2048),
    return_to: str | None = Query(default=None, max_length=2048),
) -> OAuthAuthorizeRead:
    authorize_url, nonce = build_oauth_authorize_url(
        provider,
        redirect_uri=redirect_uri,
        return_to=return_to,
    )
    response.set_cookie(
        _OAUTH_NONCE_COOKIE,
        nonce,
        httponly=True,
        samesite="strict",
        secure=True,
        max_age=_OAUTH_NONCE_MAX_AGE,
    )
    return OAuthAuthorizeRead(authorize_url=authorize_url)


@router.post("/oauth/{provider}/exchange", response_model=AuthSessionRead)
async def exchange_oauth_callback(
    provider: AuthProvider,
    payload: OAuthExchangeRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> AuthSessionRead:
    nonce = request.cookies.get(_OAUTH_NONCE_COOKIE)
    user, session_token, redirect_to = await exchange_oauth_code(
        db,
        provider=provider,
        code=payload.code,
        state=payload.state,
        redirect_uri=payload.redirect_uri,
        nonce=nonce,
    )
    response.delete_cookie(_OAUTH_NONCE_COOKIE)
    return _build_auth_response(
        user=user,
        session_token=session_token,
        redirect_to=redirect_to,
    )
