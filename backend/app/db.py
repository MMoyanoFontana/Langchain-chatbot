from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401
    from app.services.catalog import seed_provider_catalog

    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        # Remove legacy first-class Ollama provider rows if present.
        # This keeps Enum decoding stable after ollama was moved back to generic `other`.
        legacy_provider_ids = [
            row[0]
            for row in conn.execute(
                text("SELECT id FROM providers WHERE code IN ('OLLAMA', 'ollama')")
            ).fetchall()
        ]
        for provider_id in legacy_provider_ids:
            conn.execute(
                text("UPDATE chat_messages SET provider_id = NULL WHERE provider_id = :provider_id"),
                {"provider_id": provider_id},
            )
            conn.execute(
                text(
                    "UPDATE chat_threads SET provider_api_key_id = NULL "
                    "WHERE provider_api_key_id IN ("
                    "SELECT id FROM provider_api_keys WHERE provider_id = :provider_id"
                    ")"
                ),
                {"provider_id": provider_id},
            )
            conn.execute(
                text("DELETE FROM provider_models WHERE provider_id = :provider_id"),
                {"provider_id": provider_id},
            )
            conn.execute(
                text("DELETE FROM provider_api_keys WHERE provider_id = :provider_id"),
                {"provider_id": provider_id},
            )
            conn.execute(
                text("DELETE FROM providers WHERE id = :provider_id"),
                {"provider_id": provider_id},
            )
    with SessionLocal() as db:
        seed_provider_catalog(db)
