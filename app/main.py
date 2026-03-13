from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.schemas.models import HealthResponse
from app.dependencies import get_settings
from app.api import routes_index, routes_ask, routes_sources
from app.core.rate_limit import limiter
from app.core.logging_config import configure_logging_from_settings
from app.middleware.logging_middleware import RequestLoggingMiddleware

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


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=get_settings().app_version)
