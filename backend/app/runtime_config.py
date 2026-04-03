from __future__ import annotations

import os
from functools import lru_cache

class RuntimeConfigError(RuntimeError):
    pass


def _parse_cors_origins(raw_value: str) -> list[str]:
    origins: list[str] = []
    for entry in raw_value.split(","):
        normalized = entry.strip().rstrip("/")
        if normalized:
            origins.append(normalized)
    return list(dict.fromkeys(origins))



@lru_cache(maxsize=1)
def get_cors_allowed_origins() -> list[str]:
    configured_origins = (
        os.getenv("BACKEND_CORS_ORIGINS", "").strip()
        or os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    )
    if not configured_origins:
        raise RuntimeConfigError(
            "BACKEND_CORS_ORIGINS must be configured. Set it to a comma-separated list of allowed frontend origins."
        )

    origins = _parse_cors_origins(configured_origins)
    if not origins:
        raise RuntimeConfigError(
            "BACKEND_CORS_ORIGINS must contain at least one valid origin."
        )
    return origins
