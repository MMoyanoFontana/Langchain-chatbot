from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload, selectinload

from app.db import get_db
from app.models import ChatMessage, ChatThread, DocumentIndexStatus, IndexedDocument, Provider, ProviderApiKey, ProviderCode, User, UserMemory, utc_now
from app.schemas import (
    ChatMessageRead,
    ChatThreadDocumentRead,
    ChatThreadRead,
    ChatThreadSummaryRead,
    ChatThreadUpdate,
    ProviderApiKeyRead,
    ProviderApiKeyUpdate,
    ProviderApiKeyUpsert,
    ProviderRead,
    ProviderSettingsRead,
    UserMemoryRead,
    UserMemoryUpdate,
    UserRead,
    UserUpdate,
)
from app.security import EncryptionConfigError, decrypt_secret, encrypt_secret, mask_secret
from app.services.current_user import require_current_user
from app.services.rag import RagConfigurationError, get_rag_service

router = APIRouter(prefix="/users", tags=["users"])


def _get_provider_or_404(db: Session, provider_code: ProviderCode) -> Provider:
    provider = db.scalar(select(Provider).where(Provider.code == provider_code))
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found.")
    return provider


def _get_thread_or_404(db: Session, user_id: str, thread_id: str, include_messages: bool = False) -> ChatThread:
    query = select(ChatThread).where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
    if include_messages:
        query = query.options(
            selectinload(ChatThread.messages).selectinload(ChatMessage.provider),
            selectinload(ChatThread.indexed_documents),
        )

    thread = db.scalar(query)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat thread not found.")
    return thread


def _normalize_key_name(key_name: str) -> str:
    normalized = key_name.strip()
    if not normalized:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="`key_name` cannot be blank.")
    return normalized


def _build_api_key_response(api_key: ProviderApiKey) -> ProviderApiKeyRead:
    if api_key.provider is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing provider relation.")

    try:
        masked_value = mask_secret(decrypt_secret(api_key.encrypted_api_key))
    except (EncryptionConfigError, ValueError):
        masked_value = "********"

    return ProviderApiKeyRead(
        id=api_key.id,
        user_id=api_key.user_id,
        key_name=api_key.key_name,
        is_default=api_key.is_default,
        is_active=api_key.is_active,
        masked_api_key=masked_value,
        provider=ProviderRead.model_validate(api_key.provider),
        created_at=api_key.created_at,
        updated_at=api_key.updated_at,
    )


def _build_thread_read(thread: ChatThread) -> ChatThreadRead:
    documents = sorted(
        thread.indexed_documents,
        key=lambda document: document.created_at,
        reverse=True,
    )
    return ChatThreadRead(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        documents=[ChatThreadDocumentRead.model_validate(document) for document in documents],
        messages=[ChatMessageRead.model_validate(message) for message in thread.messages],
    )


def _update_user(user: User, payload: UserUpdate, db: Session) -> User:
    updates = payload.model_dump(exclude_unset=True)

    if "email" in updates and updates["email"] is not None:
        normalized_email = updates["email"].strip().lower()
        if not normalized_email:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email cannot be blank.")
        email_owner = db.scalar(select(User).where(User.email == normalized_email, User.id != user.id))
        if email_owner is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Another user already has this email.")
        user.email = normalized_email

    if "full_name" in updates:
        user.full_name = updates["full_name"]
    if "is_active" in updates and updates["is_active"] is not None:
        user.is_active = updates["is_active"]

    db.commit()
    db.refresh(user)
    return user


def list_user_threads(user_id: str, db: Session) -> list[ChatThread]:
    threads = db.scalars(
        select(ChatThread)
        .where(ChatThread.user_id == user_id)
        .order_by(ChatThread.updated_at.desc())
    ).all()
    return list(threads)


def get_user_thread(user_id: str, thread_id: str, db: Session) -> ChatThreadRead:
    thread = _get_thread_or_404(db, user_id, thread_id, include_messages=True)
    return _build_thread_read(thread)


def update_user_thread(
    user_id: str,
    thread_id: str,
    payload: ChatThreadUpdate,
    db: Session,
) -> ChatThread:
    thread = _get_thread_or_404(db, user_id, thread_id)
    normalized_title = payload.title.strip()
    if not normalized_title:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Thread title cannot be blank.")

    thread.title = normalized_title
    db.commit()
    db.refresh(thread)
    return thread


def delete_user_thread(user_id: str, thread_id: str, db: Session) -> Response:
    thread = _get_thread_or_404(db, user_id, thread_id)
    asyncio.run(
        get_rag_service().delete_thread_documents(
            db=db,
            user_id=user_id,
            thread_id=thread_id,
        )
    )
    db.delete(thread)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def list_user_provider_settings(user_id: str, db: Session) -> list[ProviderSettingsRead]:
    providers = list(db.scalars(select(Provider).order_by(Provider.display_name.asc())).all())
    keys = list(db.scalars(select(ProviderApiKey).where(ProviderApiKey.user_id == user_id)).all())

    keys_by_provider: dict[int, list[ProviderApiKey]] = {}
    for key in keys:
        keys_by_provider.setdefault(key.provider_id, []).append(key)

    response: list[ProviderSettingsRead] = []
    for provider in providers:
        provider_keys = keys_by_provider.get(provider.id, [])
        default_key = next((key for key in provider_keys if key.is_default), None)
        response.append(
            ProviderSettingsRead(
                provider=ProviderRead.model_validate(provider),
                has_key=len(provider_keys) > 0,
                default_key_name=default_key.key_name if default_key else None,
            )
        )
    return response


def list_user_api_keys(user_id: str, db: Session) -> list[ProviderApiKeyRead]:
    api_keys = list(
        db.scalars(
            select(ProviderApiKey)
            .where(ProviderApiKey.user_id == user_id)
            .options(joinedload(ProviderApiKey.provider))
            .order_by(ProviderApiKey.created_at.asc())
        ).all()
    )
    return [_build_api_key_response(api_key) for api_key in api_keys]


def upsert_user_api_key(
    user_id: str,
    provider_code: ProviderCode,
    payload: ProviderApiKeyUpsert,
    db: Session,
) -> ProviderApiKeyRead:
    provider = _get_provider_or_404(db, provider_code)
    key_name = _normalize_key_name(payload.key_name)

    try:
        encrypted_api_key = encrypt_secret(payload.api_key)
    except EncryptionConfigError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    api_key = db.scalar(
        select(ProviderApiKey).where(
            ProviderApiKey.user_id == user_id,
            ProviderApiKey.provider_id == provider.id,
            ProviderApiKey.key_name == key_name,
        )
    )
    if api_key is None:
        api_key = ProviderApiKey(
            user_id=user_id,
            provider_id=provider.id,
            key_name=key_name,
            encrypted_api_key=encrypted_api_key,
            is_default=payload.is_default,
            is_active=payload.is_active,
        )
        db.add(api_key)
        db.flush()
    else:
        api_key.encrypted_api_key = encrypted_api_key
        api_key.is_default = payload.is_default
        api_key.is_active = payload.is_active

    if payload.is_default:
        db.execute(
            update(ProviderApiKey)
            .where(
                ProviderApiKey.user_id == user_id,
                ProviderApiKey.provider_id == provider.id,
                ProviderApiKey.id != api_key.id,
            )
            .values(is_default=False)
        )

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Unable to store provider API key.") from exc

    stored_api_key = db.scalar(
        select(ProviderApiKey)
        .where(ProviderApiKey.id == api_key.id)
        .options(joinedload(ProviderApiKey.provider))
    )
    if stored_api_key is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key not found after save.")

    return _build_api_key_response(stored_api_key)


def update_user_api_key(
    user_id: str,
    api_key_id: str,
    payload: ProviderApiKeyUpdate,
    db: Session,
) -> ProviderApiKeyRead:
    api_key = db.scalar(
        select(ProviderApiKey)
        .where(ProviderApiKey.id == api_key_id, ProviderApiKey.user_id == user_id)
        .options(joinedload(ProviderApiKey.provider))
    )
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider API key not found.")

    updates = payload.model_dump(exclude_unset=True)
    if "key_name" in updates and updates["key_name"] is not None:
        api_key.key_name = _normalize_key_name(updates["key_name"])

    if "api_key" in updates and updates["api_key"] is not None:
        try:
            api_key.encrypted_api_key = encrypt_secret(updates["api_key"])
        except EncryptionConfigError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    if "is_active" in updates and updates["is_active"] is not None:
        api_key.is_active = updates["is_active"]

    if "is_default" in updates and updates["is_default"] is not None:
        api_key.is_default = updates["is_default"]
        if updates["is_default"]:
            db.execute(
                update(ProviderApiKey)
                .where(
                    ProviderApiKey.user_id == user_id,
                    ProviderApiKey.provider_id == api_key.provider_id,
                    ProviderApiKey.id != api_key.id,
                )
                .values(is_default=False)
            )

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Unable to update provider API key.") from exc

    db.refresh(api_key)
    refreshed_api_key = db.scalar(
        select(ProviderApiKey)
        .where(ProviderApiKey.id == api_key.id)
        .options(joinedload(ProviderApiKey.provider))
    )
    if refreshed_api_key is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Provider API key not found.")

    return _build_api_key_response(refreshed_api_key)


def delete_user_thread_document(
    user_id: str,
    thread_id: str,
    document_id: str,
    db: Session,
) -> Response:
    _get_thread_or_404(db, user_id, thread_id)
    document = db.scalar(
        select(IndexedDocument).where(
            IndexedDocument.id == document_id,
            IndexedDocument.thread_id == thread_id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    asyncio.run(
        get_rag_service().delete_document(
            db=db,
            user_id=user_id,
            thread_id=thread_id,
            document_id=document_id,
        )
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def retry_user_thread_document(
    user_id: str,
    thread_id: str,
    document_id: str,
    db: Session,
) -> ChatThreadDocumentRead:
    _get_thread_or_404(db, user_id, thread_id)
    document = db.scalar(
        select(IndexedDocument).where(
            IndexedDocument.id == document_id,
            IndexedDocument.thread_id == thread_id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if document.status != DocumentIndexStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only failed documents can be retried.",
        )

    try:
        updated = asyncio.run(
            get_rag_service().retry_document(
                db=db,
                user_id=user_id,
                thread_id=thread_id,
                document_id=document_id,
            )
        )
    except RagConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return ChatThreadDocumentRead.model_validate(updated)


def delete_user_api_key(user_id: str, api_key_id: str, db: Session) -> Response:
    api_key = db.scalar(
        select(ProviderApiKey).where(ProviderApiKey.id == api_key_id, ProviderApiKey.user_id == user_id)
    )
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider API key not found.")

    db.delete(api_key)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserRead)
def get_current_user_profile(user: User = Depends(require_current_user)) -> User:
    return user


@router.patch("/me", response_model=UserRead)
def update_current_user_profile(
    payload: UserUpdate,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> User:
    return _update_user(user, payload, db)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_user_profile(
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> Response:
    asyncio.run(
        get_rag_service().delete_user_documents(
            db=db,
            user_id=user.id,
        )
    )
    db.delete(user)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me/settings/providers", response_model=list[ProviderSettingsRead])
def list_current_user_provider_settings(
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> list[ProviderSettingsRead]:
    return list_user_provider_settings(user.id, db)


@router.get("/me/settings/api-keys", response_model=list[ProviderApiKeyRead])
def list_current_user_api_keys(
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> list[ProviderApiKeyRead]:
    return list_user_api_keys(user.id, db)


@router.put("/me/settings/api-keys/{provider_code}", response_model=ProviderApiKeyRead)
def upsert_current_user_api_key(
    provider_code: ProviderCode,
    payload: ProviderApiKeyUpsert,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> ProviderApiKeyRead:
    return upsert_user_api_key(user.id, provider_code, payload, db)


@router.patch("/me/settings/api-keys/{api_key_id}", response_model=ProviderApiKeyRead)
def update_current_user_api_key(
    api_key_id: str,
    payload: ProviderApiKeyUpdate,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> ProviderApiKeyRead:
    return update_user_api_key(user.id, api_key_id, payload, db)


@router.delete("/me/settings/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_user_api_key(
    api_key_id: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> Response:
    return delete_user_api_key(user.id, api_key_id, db)


@router.get("/me/threads", response_model=list[ChatThreadSummaryRead])
def list_current_user_threads(
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> list[ChatThread]:
    return list_user_threads(user.id, db)


@router.get("/me/threads/{thread_id}", response_model=ChatThreadRead)
def get_current_user_thread(
    thread_id: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> ChatThreadRead:
    return get_user_thread(user.id, thread_id, db)


@router.patch("/me/threads/{thread_id}", response_model=ChatThreadSummaryRead)
def update_current_user_thread(
    thread_id: str,
    payload: ChatThreadUpdate,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> ChatThread:
    return update_user_thread(user.id, thread_id, payload, db)


@router.delete("/me/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_user_thread(
    thread_id: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> Response:
    return delete_user_thread(user.id, thread_id, db)


@router.delete("/me/threads/{thread_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_user_thread_document(
    thread_id: str,
    document_id: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> Response:
    return delete_user_thread_document(user.id, thread_id, document_id, db)


@router.post("/me/threads/{thread_id}/documents/{document_id}/retry", response_model=ChatThreadDocumentRead)
def retry_current_user_thread_document(
    thread_id: str,
    document_id: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> ChatThreadDocumentRead:
    return retry_user_thread_document(user.id, thread_id, document_id, db)


# ---------------------------------------------------------------------------
# Message management
# ---------------------------------------------------------------------------

def delete_single_message(user_id: str, thread_id: str, message_id: str, db: Session) -> None:
    thread = db.scalar(
        select(ChatThread).where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
    )
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")

    message = db.scalar(
        select(ChatMessage).where(
            ChatMessage.id == message_id, ChatMessage.thread_id == thread_id
        )
    )
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found.")

    db.delete(message)
    thread.updated_at = utc_now()
    db.commit()


def delete_messages_from(user_id: str, thread_id: str, message_id: str, db: Session) -> None:
    thread = db.scalar(
        select(ChatThread).where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
    )
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")

    pivot = db.scalar(
        select(ChatMessage).where(
            ChatMessage.id == message_id, ChatMessage.thread_id == thread_id
        )
    )
    if pivot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found.")

    to_delete = list(
        db.scalars(
            select(ChatMessage)
            .where(
                ChatMessage.thread_id == thread_id,
                ChatMessage.created_at >= pivot.created_at,
            )
            .order_by(ChatMessage.created_at)
        )
    )
    for msg in to_delete:
        db.delete(msg)
    thread.updated_at = utc_now()
    db.commit()


@router.delete(
    "/me/threads/{thread_id}/messages/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_current_user_message(
    thread_id: str,
    message_id: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> Response:
    delete_single_message(user.id, thread_id, message_id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/me/threads/{thread_id}/messages/{message_id}/from-here",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_current_user_messages_from(
    thread_id: str,
    message_id: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> Response:
    delete_messages_from(user.id, thread_id, message_id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# User memory management
# ---------------------------------------------------------------------------

def list_user_memories(user_id: str, db: Session) -> list[UserMemory]:
    return list(
        db.scalars(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.updated_at.desc())
            .limit(100)
        )
    )


def update_user_memory(user_id: str, key: str, payload: UserMemoryUpdate, db: Session) -> UserMemory:
    memory = db.scalar(
        select(UserMemory).where(
            UserMemory.user_id == user_id,
            UserMemory.key == key,
        )
    )
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory fact not found.")
    memory.value = payload.value
    memory.updated_at = utc_now()
    db.commit()
    db.refresh(memory)
    return memory


def delete_user_memory(user_id: str, key: str, db: Session) -> None:
    memory = db.scalar(
        select(UserMemory).where(
            UserMemory.user_id == user_id,
            UserMemory.key == key,
        )
    )
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory fact not found.")
    db.delete(memory)
    db.commit()


@router.get("/me/memories", response_model=list[UserMemoryRead])
def list_current_user_memories(
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> list[UserMemory]:
    return list_user_memories(user.id, db)


@router.patch("/me/memories/{key}", response_model=UserMemoryRead)
def update_current_user_memory(
    key: str,
    payload: UserMemoryUpdate,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> UserMemory:
    return update_user_memory(user.id, key, payload, db)


@router.delete("/me/memories/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_user_memory(
    key: str,
    user: User = Depends(require_current_user),
    db: Session = Depends(get_db),
) -> Response:
    delete_user_memory(user.id, key, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
