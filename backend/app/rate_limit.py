from __future__ import annotations

import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.current_user import get_session_token_from_request


def _rate_limit_key(request: Request) -> str:
    token = get_session_token_from_request(request)
    if token:
        return f"token:{token}"
    return get_remote_address(request)


def get_rate_limit() -> str:
    """Callable limit accepted by slowapi — evaluated per request so the env
    var can be changed without restarting the process."""
    per_minute = os.getenv("RATE_LIMIT_PER_MINUTE", "60")
    return f"{per_minute}/minute"


limiter = Limiter(key_func=_rate_limit_key)
