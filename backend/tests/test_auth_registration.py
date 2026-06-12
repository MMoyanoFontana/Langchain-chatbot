from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base
from app.models import AuthSession, User
from app.services.auth import (
    hash_password,
    login_user_with_username,
    register_user_with_username,
    verify_password,
)


def _session_factory():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def test_hash_password_round_trip() -> None:
    password_hash = hash_password("password123")

    assert password_hash.startswith("scrypt$")
    assert verify_password("password123", password_hash)
    assert not verify_password("wrong-password", password_hash)


def test_register_user_with_username_creates_user_and_session() -> None:
    SessionLocal = _session_factory()

    with SessionLocal() as db:
        user, session_token = register_user_with_username(
            db,
            username="Signup.User",
            password="password123",
            full_name="Signup User",
        )

        persisted_user = db.scalar(select(User).where(User.id == user.id))
        persisted_session = db.scalar(
            select(AuthSession).where(AuthSession.user_id == user.id)
        )

        assert session_token
        assert persisted_user is not None
        assert persisted_user.username == "signup.user"
        assert persisted_user.full_name == "Signup User"
        assert persisted_user.password_hash is not None
        assert persisted_session is not None


def test_register_user_with_username_rejects_duplicates() -> None:
    SessionLocal = _session_factory()

    with SessionLocal() as db:
        register_user_with_username(
            db, username="taken", password="password123", full_name=None
        )

        with pytest.raises(HTTPException) as exc_info:
            register_user_with_username(
                db, username="TAKEN", password="other-password", full_name=None
            )

        assert exc_info.value.status_code == 409


def test_login_user_with_username_round_trip() -> None:
    SessionLocal = _session_factory()

    with SessionLocal() as db:
        registered, _ = register_user_with_username(
            db, username="login.user", password="password123", full_name=None
        )

        user, session_token = login_user_with_username(
            db, username="login.user", password="password123"
        )

        assert user.id == registered.id
        assert session_token

        with pytest.raises(HTTPException) as exc_info:
            login_user_with_username(
                db, username="login.user", password="wrong-password"
            )

        assert exc_info.value.status_code == 401
