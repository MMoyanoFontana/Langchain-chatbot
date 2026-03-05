from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User

DEV_USER_ID = "00000000-0000-0000-0000-000000000001"
DEV_USER_EMAIL = "user@local.chatbot"


def get_dev_user(db: Session) -> User | None:
    user = db.get(User, DEV_USER_ID)
    if user is not None:
        return user
    return db.scalar(select(User).where(User.email == DEV_USER_EMAIL))


def get_dev_user_or_404(db: Session) -> User:
    user = get_dev_user(db)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Current user not found.")
    return user
