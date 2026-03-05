from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload, selectinload

from app.db import get_db
from app.models import ChatMessage, ChatThread, Provider, ProviderApiKey, ProviderCode, User
from app.schemas import (
    ChatThreadRead,
    ChatThreadSummaryRead,
    ChatThreadUpdate,
    ProviderApiKeyRead,
    ProviderApiKeyUpdate,
    ProviderApiKeyUpsert,
    ProviderRead,
    ProviderSettingsRead,
    UserCreate,
    UserRead,
    UserUpdate,
)
from app.services.current_user import get_dev_user, get_dev_user_or_404
from app.security import EncryptionConfigError, decrypt_secret, encrypt_secret, mask_secret

router = APIRouter(prefix="/users", tags=["users"])


def _get_user_or_404(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user


def _get_provider_or_404(db: Session, provider_code: ProviderCode) -> Provider:
    provider = db.scalar(select(Provider).where(Provider.code == provider_code))
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found.")
    return provider


def _get_thread_or_404(db: Session, user_id: str, thread_id: str, include_messages: bool = False) -> ChatThread:
    query = select(ChatThread).where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
    if include_messages:
        query = query.options(selectinload(ChatThread.messages).selectinload(ChatMessage.provider))

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
    return ChatThreadRead.model_validate(thread)


@router.get("", response_model=list[UserRead])
def list_users(db: Session = Depends(get_db)) -> list[User]:
    return list(db.scalars(select(User).order_by(User.created_at.desc())).all())


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email cannot be blank.")

    existing = db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists.")

    user = User(email=email, full_name=payload.full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/dev/current", response_model=UserRead)
def get_dev_current_user(db: Session = Depends(get_db)) -> User:
    return get_dev_user_or_404(db)


@router.patch("/dev/current", response_model=UserRead)
def update_dev_current_user(payload: UserUpdate, db: Session = Depends(get_db)) -> User:
    user = get_dev_user_or_404(db)
    return update_user(user.id, payload, db)


@router.delete("/dev/current", status_code=status.HTTP_204_NO_CONTENT)
def delete_dev_current_user(db: Session = Depends(get_db)) -> Response:
    user = get_dev_user(db)
    if user is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    db.delete(user)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/dev/current/settings/providers", response_model=list[ProviderSettingsRead])
def list_dev_user_provider_settings(db: Session = Depends(get_db)) -> list[ProviderSettingsRead]:
    user = get_dev_user_or_404(db)
    return list_user_provider_settings(user.id, db)


@router.get("/dev/current/settings/api-keys", response_model=list[ProviderApiKeyRead])
def list_dev_user_api_keys(db: Session = Depends(get_db)) -> list[ProviderApiKeyRead]:
    user = get_dev_user_or_404(db)
    return list_user_api_keys(user.id, db)


@router.put("/dev/current/settings/api-keys/{provider_code}", response_model=ProviderApiKeyRead)
def upsert_dev_user_api_key(
    provider_code: ProviderCode,
    payload: ProviderApiKeyUpsert,
    db: Session = Depends(get_db),
) -> ProviderApiKeyRead:
    user = get_dev_user_or_404(db)
    return upsert_user_api_key(user.id, provider_code, payload, db)


@router.patch("/dev/current/settings/api-keys/{api_key_id}", response_model=ProviderApiKeyRead)
def update_dev_user_api_key(
    api_key_id: str,
    payload: ProviderApiKeyUpdate,
    db: Session = Depends(get_db),
) -> ProviderApiKeyRead:
    user = get_dev_user_or_404(db)
    return update_user_api_key(user.id, api_key_id, payload, db)


@router.delete("/dev/current/settings/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dev_user_api_key(api_key_id: str, db: Session = Depends(get_db)) -> Response:
    user = get_dev_user_or_404(db)
    return delete_user_api_key(user.id, api_key_id, db)


@router.get("/dev/current/threads", response_model=list[ChatThreadSummaryRead])
def list_dev_user_threads(db: Session = Depends(get_db)) -> list[ChatThread]:
    user = get_dev_user_or_404(db)
    return list_user_threads(user.id, db)


@router.get("/dev/current/threads/{thread_id}", response_model=ChatThreadRead)
def get_dev_user_thread(thread_id: str, db: Session = Depends(get_db)) -> ChatThreadRead:
    user = get_dev_user_or_404(db)
    return get_user_thread(user.id, thread_id, db)


@router.patch("/dev/current/threads/{thread_id}", response_model=ChatThreadSummaryRead)
def update_dev_user_thread(
    thread_id: str,
    payload: ChatThreadUpdate,
    db: Session = Depends(get_db),
) -> ChatThread:
    user = get_dev_user_or_404(db)
    return update_user_thread(user.id, thread_id, payload, db)


@router.delete("/dev/current/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dev_user_thread(thread_id: str, db: Session = Depends(get_db)) -> Response:
    user = get_dev_user_or_404(db)
    return delete_user_thread(user.id, thread_id, db)


@router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: str, db: Session = Depends(get_db)) -> User:
    return _get_user_or_404(db, user_id)


@router.patch("/{user_id}", response_model=UserRead)
def update_user(user_id: str, payload: UserUpdate, db: Session = Depends(get_db)) -> User:
    user = _get_user_or_404(db, user_id)
    updates = payload.model_dump(exclude_unset=True)

    if "email" in updates and updates["email"] is not None:
        normalized_email = updates["email"].strip().lower()
        if not normalized_email:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email cannot be blank.")
        email_owner = db.scalar(select(User).where(User.email == normalized_email, User.id != user_id))
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


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: str, db: Session = Depends(get_db)) -> Response:
    user = _get_user_or_404(db, user_id)
    db.delete(user)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{user_id}/threads", response_model=list[ChatThreadSummaryRead])
def list_user_threads(user_id: str, db: Session = Depends(get_db)) -> list[ChatThread]:
    _get_user_or_404(db, user_id)
    threads = db.scalars(
        select(ChatThread)
        .where(ChatThread.user_id == user_id)
        .order_by(ChatThread.updated_at.desc())
    ).all()
    return list(threads)


@router.get("/{user_id}/threads/{thread_id}", response_model=ChatThreadRead)
def get_user_thread(user_id: str, thread_id: str, db: Session = Depends(get_db)) -> ChatThreadRead:
    _get_user_or_404(db, user_id)
    thread = _get_thread_or_404(db, user_id, thread_id, include_messages=True)
    return _build_thread_read(thread)


@router.patch("/{user_id}/threads/{thread_id}", response_model=ChatThreadSummaryRead)
def update_user_thread(
    user_id: str,
    thread_id: str,
    payload: ChatThreadUpdate,
    db: Session = Depends(get_db),
) -> ChatThread:
    _get_user_or_404(db, user_id)
    thread = _get_thread_or_404(db, user_id, thread_id)
    normalized_title = payload.title.strip()
    if not normalized_title:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Thread title cannot be blank.")

    thread.title = normalized_title
    db.commit()
    db.refresh(thread)
    return thread


@router.delete("/{user_id}/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_thread(user_id: str, thread_id: str, db: Session = Depends(get_db)) -> Response:
    _get_user_or_404(db, user_id)
    thread = _get_thread_or_404(db, user_id, thread_id)
    db.delete(thread)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{user_id}/settings/providers", response_model=list[ProviderSettingsRead])
def list_user_provider_settings(user_id: str, db: Session = Depends(get_db)) -> list[ProviderSettingsRead]:
    _get_user_or_404(db, user_id)

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


@router.get("/{user_id}/settings/api-keys", response_model=list[ProviderApiKeyRead])
def list_user_api_keys(user_id: str, db: Session = Depends(get_db)) -> list[ProviderApiKeyRead]:
    _get_user_or_404(db, user_id)

    api_keys = list(
        db.scalars(
            select(ProviderApiKey)
            .where(ProviderApiKey.user_id == user_id)
            .options(joinedload(ProviderApiKey.provider))
            .order_by(ProviderApiKey.created_at.asc())
        ).all()
    )
    return [_build_api_key_response(api_key) for api_key in api_keys]


@router.put("/{user_id}/settings/api-keys/{provider_code}", response_model=ProviderApiKeyRead)
def upsert_user_api_key(
    user_id: str,
    provider_code: ProviderCode,
    payload: ProviderApiKeyUpsert,
    db: Session = Depends(get_db),
) -> ProviderApiKeyRead:
    _get_user_or_404(db, user_id)
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


@router.patch("/{user_id}/settings/api-keys/{api_key_id}", response_model=ProviderApiKeyRead)
def update_user_api_key(
    user_id: str,
    api_key_id: str,
    payload: ProviderApiKeyUpdate,
    db: Session = Depends(get_db),
) -> ProviderApiKeyRead:
    _get_user_or_404(db, user_id)

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


@router.delete("/{user_id}/settings/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_api_key(user_id: str, api_key_id: str, db: Session = Depends(get_db)) -> Response:
    _get_user_or_404(db, user_id)

    api_key = db.scalar(
        select(ProviderApiKey).where(ProviderApiKey.id == api_key_id, ProviderApiKey.user_id == user_id)
    )
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider API key not found.")

    db.delete(api_key)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
