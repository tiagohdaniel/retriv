from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.dependencies import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    settings = get_settings()
    if not settings.api_auth_enabled:
        return
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
