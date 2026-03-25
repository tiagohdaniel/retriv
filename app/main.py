import base64
import os
import secrets
import traceback
from contextlib import asynccontextmanager

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
from app.api import routes_index, routes_ask, routes_sources, routes_analyze
from app.mcp.server import mcp
from app.core.rate_limit import limiter
from app.core.logging_config import configure_logging_from_settings
from app.middleware.logging_middleware import RequestLoggingMiddleware

logger = structlog.get_logger(__name__)

configure_logging_from_settings()

_settings = get_settings()


def _maybe_clear_collection() -> None:
    if os.environ.get("CLEAR_COLLECTION_ON_STARTUP", "").lower() != "true":
        return
    try:
        from app.dependencies import get_chroma_client
        client = get_chroma_client()
        client.delete_collection(_settings.chroma_collection)
        logger.info("collection_cleared", collection=_settings.chroma_collection)
    except Exception as exc:
        logger.warning("collection_clear_failed", error=str(exc))


_maybe_clear_collection()

# MCP HTTP app must be created before FastAPI so its lifespan can be wired in
_mcp_http_app = mcp.http_app()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    async with _mcp_http_app.lifespan(app):
        yield


_cors_origins = (
    ["*"]
    if _settings.cors_origins.strip() == "*"
    else [o.strip() for o in _settings.cors_origins.split(",")]
)

_tags_metadata = [
    {
        "name": "indexing",
        "description": (
            "Ingest documents into the vector store. "
            "Indexing is **asynchronous** — POST `/index` returns a `job_id` immediately; "
            "poll `/index/status/{job_id}` to track progress. "
            "Documents are semantically chunked, batch-embedded via ONNX, and upserted into ChromaDB."
        ),
    },
    {
        "name": "search",
        "description": (
            "Query indexed documents with natural language. "
            "Uses **hybrid retrieval** (BM25 + dense semantic search) merged via Reciprocal Rank Fusion, "
            "optionally reranked by a cross-encoder model before being sent to the LLM. "
            "Supports both blocking JSON and **SSE streaming** responses."
        ),
    },
    {
        "name": "sources",
        "description": "List and delete indexed sources. Deletions are scoped to the authenticated tenant.",
    },
    {
        "name": "analytics",
        "description": (
            "Run structured analysis over tabular data (CSV/Excel). "
            "Deterministic aggregations via pandas; LLM formats the result — never calculates."
        ),
    },
    {
        "name": "system",
        "description": "Health check and Prometheus metrics.",
    },
]

app = FastAPI(
    lifespan=_lifespan,
    title="retriv",
    description=(
        "**Production RAG engine — multi-tenant, domain-agnostic.**\n\n"
        "retriv lets you index any text content and query it with natural language. "
        "Answers are grounded on retrieved document chunks with source references.\n\n"
        "### Pipeline\n"
        "```\n"
        "POST /index  →  semantic chunking  →  ONNX batch embedding  →  ChromaDB upsert\n"
        "POST /ask    →  hybrid retrieval (BM25 + dense)  →  RRF merge  →  reranker  →  LLM\n"
        "```\n\n"
        "### Key features\n"
        "- **Hybrid search** — BM25 keyword + semantic dense retrieval merged via RRF\n"
        "- **Cross-encoder reranker** — second-pass scoring for precision\n"
        "- **Semantic chunking** — paragraph-aware splitting, not fixed-size windows\n"
        "- **Multi-tenancy** — each API key maps to an isolated tenant; zero data bleed\n"
        "- **MCP server** — AI agents (AGNO, LangGraph, Claude Desktop) connect at `/mcp`\n"
        "- **SSE streaming** — token-by-token delivery via `/ask/stream`\n"
        "- **No GPU required** — embeddings run via ONNX Runtime\n\n"
        "### Authentication\n"
        "Pass your API key in the `X-API-Key` header on every request."
    ),
    version=_settings.app_version,
    openapi_tags=_tags_metadata,
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
app.include_router(routes_analyze.router, tags=["analytics"])

# MCP server — Streamable HTTP transport (MCP spec 2025-03-26)
# External agents (AGNO, Claude Desktop, LangGraph, etc.) connect at /mcp/mcp
# using the same API key auth as the REST endpoints.
# Note: FastMCP's http_app() routes to /mcp internally, so the full path
# when mounted at /mcp becomes /mcp/mcp.
app.mount("/mcp", _mcp_http_app)

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
