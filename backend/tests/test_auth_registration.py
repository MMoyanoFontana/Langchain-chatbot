from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base
from app.models import AuthSession, User
from app.services.auth import hash_password, register_user_with_password, verify_password


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


def test_register_user_with_password_creates_user_and_session() -> None:
    SessionLocal = _session_factory()

    with SessionLocal() as db:
        user, session_token = register_user_with_password(
            db,
            email="signup@example.com",
            password="password123",
            full_name="Signup User",
        )

        persisted_user = db.scalar(select(User).where(User.id == user.id))
        persisted_session = db.scalar(
            select(AuthSession).where(AuthSession.user_id == user.id)
        )

        assert session_token
        assert persisted_user is not None
        assert persisted_user.email == "signup@example.com"
        assert persisted_user.full_name == "Signup User"
        assert persisted_user.password_hash is not None
        assert persisted_session is not None
