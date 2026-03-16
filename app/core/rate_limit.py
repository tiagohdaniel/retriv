from fastapi import Request  # needed by _key_func
from slowapi import Limiter
from slowapi.util import get_remote_address


def _key_func(request: Request) -> str:
    """Use API key as rate limit identity when provided, otherwise fall back to IP.

    Per-key limiting is fairer in multi-tenant setups: users behind the same
    NAT/proxy are not penalised for each other's usage.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func)


def _build_limit(settings_attr: str) -> callable:
    """Return a callable that slowapi calls per request to resolve the limit string.

    slowapi inspects the callable's signature: if it has no parameters, it is
    called with no arguments. Settings are cached via lru_cache so the lookup
    is effectively free.

    When RATE_LIMIT_ENABLED=false (default for local dev) the limit is set to
    an effectively unlimited value so no tracking overhead is incurred.
    """
    def _limit() -> str:
        from app.dependencies import get_settings
        s = get_settings()
        if not s.rate_limit_enabled:
            return "10000/second"
        return getattr(s, settings_attr)
    return _limit


index_rate_limit = _build_limit("rate_limit_index")
ask_rate_limit = _build_limit("rate_limit_ask")
sources_rate_limit = _build_limit("rate_limit_sources")
analyze_rate_limit = _build_limit("rate_limit_analyze")
