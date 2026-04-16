from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.services.auth import get_user_for_session_token


def get_session_token_from_request(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "").strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()

    header_token = request.headers.get("x-session-token", "").strip()
    if header_token:
        return header_token

    return None


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User | None:
    return get_user_for_session_token(db, get_session_token_from_request(request))


def require_current_user(user: User | None = Depends(get_current_user)) -> User:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
        )
    return user


def require_admin_user(user: User = Depends(require_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user
