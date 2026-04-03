from __future__ import annotations

import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken


class EncryptionConfigError(RuntimeError):
    pass


def _get_encryption_key_bytes() -> bytes:
    configured_key = os.getenv("API_KEY_ENCRYPTION_KEY", "").strip()
    if configured_key:
        return configured_key.encode("utf-8")

    raise EncryptionConfigError(
        "API_KEY_ENCRYPTION_KEY is not set. Generate one with "
        "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"`."
    )


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    try:
        return Fernet(_get_encryption_key_bytes())
    except ValueError as exc:
        raise EncryptionConfigError(
            "API_KEY_ENCRYPTION_KEY is invalid. It must be a urlsafe base64-encoded 32-byte key."
        ) from exc


def encrypt_secret(plain_text: str) -> str:
    return _get_fernet().encrypt(plain_text.encode("utf-8")).decode("utf-8")


def decrypt_secret(cipher_text: str) -> str:
    try:
        return _get_fernet().decrypt(cipher_text.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Encrypted value could not be decrypted with current key.") from exc


def mask_secret(plain_text: str, visible_chars: int = 4) -> str:
    if not plain_text:
        return ""
    if len(plain_text) <= visible_chars:
        return "*" * len(plain_text)
    hidden_chars = len(plain_text) - visible_chars
    return f"{'*' * hidden_chars}{plain_text[-visible_chars:]}"
