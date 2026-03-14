from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.dependencies import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> str | None:
    """Verify the API key and return the tenant_id.

    Returns:
        None  — auth is disabled (single-tenant, no filtering)
        str   — the tenant_id mapped to the provided key

    Raises:
        HTTPException 401 — auth is enabled but key is missing or invalid
    """
    settings = get_settings()

    if not settings.api_auth_enabled:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
        )

    # Multi-tenant mode: parse "key1:tenant1,key2:tenant2"
    if settings.api_keys:
        mapping = {
            k: v
            for pair in settings.api_keys.split(",")
            if ":" in pair
            for k, v in [pair.strip().split(":", 1)]
        }
        if api_key in mapping:
            return mapping[api_key]

    # Single-tenant fallback
    if settings.api_key and api_key == settings.api_key:
        return "default"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key.",
    )
