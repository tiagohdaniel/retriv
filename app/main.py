from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.models import HealthResponse
from app.dependencies import get_settings
from app.api import routes_index, routes_ask, routes_sources

app = FastAPI(
    title="Semantic Docs API",
    description=(
        "Domain-agnostic semantic search over your documentation.\n\n"
        "Index any text content, then ask natural language questions — "
        "answers are grounded on the retrieved chunks with source references."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_index.router, tags=["indexing"])
app.include_router(routes_ask.router, tags=["search"])
app.include_router(routes_sources.router, tags=["sources"])


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=get_settings().app_version)
