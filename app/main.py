import base64
import os
import secrets
import traceback

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas.models import HealthResponse
from app.dependencies import get_settings, get_vector_store
from app.core.vector_store import VectorStoreBase
from app.api import routes_index, routes_ask, routes_sources
from app.core.rate_limit import limiter
from app.core.logging_config import configure_logging_from_settings
from app.middleware.logging_middleware import RequestLoggingMiddleware

logger = structlog.get_logger(__name__)

configure_logging_from_settings()

_settings = get_settings()
_cors_origins = (
    ["*"]
    if _settings.cors_origins.strip() == "*"
    else [o.strip() for o in _settings.cors_origins.split(",")]
)

app = FastAPI(
    title="retriv",
    description=(
        "Domain-agnostic RAG API.\n\n"
        "Index any text content, ask natural language questions — "
        "answers are grounded on retrieved chunks with source references."
    ),
    version=_settings.app_version,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        method=request.method,
        path=request.url.path,
        exc_type=type(exc).__name__,
        traceback=traceback.format_exc(),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_index.router, tags=["indexing"])
app.include_router(routes_ask.router, tags=["search"])
app.include_router(routes_sources.router, tags=["sources"])

def _with_basic_auth(asgi_app):
    """Wrap an ASGI app with Basic auth when METRICS_USERNAME/PASSWORD are set."""
    username = _settings.metrics_username
    password = _settings.metrics_password
    if not username or not password:
        return asgi_app

    expected = base64.b64encode(f"{username}:{password}".encode()).decode()

    async def _authed(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            token = auth[len("Basic "):] if auth.startswith("Basic ") else ""
            if not secrets.compare_digest(token, expected):
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"www-authenticate", b'Basic realm="metrics"'),
                        (b"content-length", b"0"),
                    ],
                })
                await send({"type": "http.response.body", "body": b""})
                return
        await asgi_app(scope, receive, send)

    return _authed


if _settings.metrics_enabled:
    _instr = Instrumentator().instrument(app)
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import CollectorRegistry, make_asgi_app
        from prometheus_client import multiprocess as _prom_multiprocess

        _registry = CollectorRegistry()
        _prom_multiprocess.MultiProcessCollector(_registry)
        app.mount("/metrics", _with_basic_auth(make_asgi_app(registry=_registry)))
    else:
        _instr.expose(app, include_in_schema=False, tags=["system"])


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(vector_store: VectorStoreBase = Depends(get_vector_store)) -> JSONResponse:
    checks: dict[str, str] = {}
    version = get_settings().app_version

    try:
        vector_store.ping()
        checks["vector_store"] = "ok"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    status_code = 200 if overall == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content=HealthResponse(status=overall, version=version, checks=checks).model_dump(),
    )
