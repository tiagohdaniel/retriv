"""MCP server — exposes retriv knowledge engine capabilities to external agents.

Mounted at /mcp in the FastAPI app. Clients connect using the MCP Streamable HTTP
transport (MCP spec 2025-03-26).

Authentication (same mechanism as the REST API):
    Authorization: Bearer <api-key>
    or
    X-API-Key: <api-key>

When API_AUTH_ENABLED=false (default for local/single-tenant), auth is skipped.

Tools:
    ask             — full RAG: retrieve + LLM answer
    search          — raw retrieval without LLM (for agents doing their own prompting)
    index_document  — index text content into the knowledge base
    list_sources    — list indexed documents for this tenant
    delete_source   — remove a source from the index
"""
from __future__ import annotations

from typing import Any

from fastmcp import FastMCP, Context

from app.core.logging_config import get_logger
from app.dependencies import (
    get_chunker,
    get_embedding_service,
    get_llm_client,
    get_observability,
    get_query_normalizer,
    get_reranker,
    get_settings,
    get_vector_store,
)
from app.schemas.models import AskRequest, ConversationMessage, IndexRequest
from app.services.ask_service import AskService
from app.services.index_service import IndexService

logger = get_logger("retriv.mcp")

mcp = FastMCP(
    "retriv",
    instructions=(
        "retriv is a domain-agnostic knowledge engine. "
        "Use `ask` to answer questions from indexed documents (full RAG pipeline). "
        "Use `search` for raw chunk retrieval without LLM processing. "
        "Use `index_document` to add content to the knowledge base. "
        "Use `list_sources` to see what has been indexed. "
        "Use `delete_source` to remove a source from the index."
    ),
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_UNAUTHENTICATED = "__unauthenticated__"


def _resolve_tenant(api_key: str) -> str | None:
    """Map an API key to a tenant_id.

    Returns:
        None               — auth disabled; single-tenant mode (tenant_id = None)
        str (tenant_id)    — valid key, resolved tenant
        _UNAUTHENTICATED   — auth enabled but key is missing or invalid
    """
    settings = get_settings()

    if not settings.api_auth_enabled:
        return None

    if not api_key:
        return _UNAUTHENTICATED

    if settings.api_keys:
        mapping = {
            k: v
            for pair in settings.api_keys.split(",")
            if ":" in pair
            for k, v in [pair.strip().split(":", 1)]
        }
        if api_key in mapping:
            return mapping[api_key]

    if settings.api_key and api_key == settings.api_key:
        return "default"

    return _UNAUTHENTICATED


async def _get_tenant_id(ctx: Context) -> str | None:
    """Extract and validate tenant from the MCP request context.

    Reads Authorization: Bearer <key>  or  X-API-Key: <key>.
    Raises ValueError on auth failure when auth is enabled.
    """
    api_key = ""
    try:
        request = ctx.request_context.request
        auth = request.headers.get("authorization", "")
        api_key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        if not api_key:
            api_key = request.headers.get("x-api-key", "")
    except (AttributeError, RuntimeError):
        pass  # non-HTTP transport or no request context

    tenant_id = _resolve_tenant(api_key)
    if tenant_id == _UNAUTHENTICATED:
        raise ValueError("Invalid or missing API key.")

    return tenant_id


# ---------------------------------------------------------------------------
# Service factory
# ---------------------------------------------------------------------------

def _ask_service() -> AskService:
    """Build AskService from singleton dependencies (all lru_cache'd)."""
    settings = get_settings()
    return AskService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        llm_client=get_llm_client(),
        observability=get_observability(),
        reranker=get_reranker(),
        reranker_top_k_fetch=settings.reranker_top_k_fetch,
        hybrid_enabled=settings.hybrid_enabled,
        hybrid_corpus_limit=settings.hybrid_bm25_corpus_limit,
        source_diversity_enabled=settings.source_diversity_enabled,
        source_diversity_max_per_source=settings.source_diversity_max_per_source,
        neighbor_expansion_enabled=settings.neighbor_expansion_enabled,
        query_normalizer=get_query_normalizer(),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def ask(
    question: str,
    ctx: Context,
    history: list[dict] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Answer a question from indexed documents.

    Runs the full RAG pipeline: semantic + BM25 hybrid retrieval, optional
    reranking, and LLM answer generation grounded on retrieved chunks.

    Args:
        question: Natural language question.
        history:  Optional prior conversation turns for multi-turn context.
                  Each item: {"role": "user"|"assistant", "content": "..."}
        top_k:    Number of chunks to include in the LLM context (default 5).

    Returns dict with:
        answer:      LLM-generated answer grounded on retrieved chunks.
        sources:     List of source references (source_id, title, excerpt, score).
        tokens_used: Total LLM tokens consumed.
        model:       Model used for generation.
    """
    tenant_id = await _get_tenant_id(ctx)

    parsed_history: list[ConversationMessage] | None = None
    if history:
        parsed_history = [
            ConversationMessage(role=m["role"], content=m["content"]) for m in history
        ]

    request = AskRequest(question=question, top_k=top_k, history=parsed_history)
    response = await _ask_service().ask(request, background_tasks=None, tenant_id=tenant_id)

    logger.info("mcp_ask", question_len=len(question), tenant_id=tenant_id)

    return {
        "answer": response.answer,
        "sources": [s.model_dump() for s in response.sources],
        "tokens_used": response.tokens_used,
        "model": response.model,
    }


@mcp.tool()
async def search(
    query: str,
    ctx: Context,
    top_k: int = 5,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve relevant document chunks without LLM processing.

    Use this when you want to supply the retrieved context to your own prompt,
    or to inspect what the knowledge base contains for a given query.

    Args:
        query:      The search query.
        top_k:      Number of chunks to return (default 5, max 20).
        source_ids: Restrict search to specific source IDs (optional).

    Returns dict with:
        chunks: List of matching chunks, each with:
                id, content, source_id, title, chunk_index, distance.
        count:  Number of chunks returned.
    """
    tenant_id = await _get_tenant_id(ctx)

    request = AskRequest(question=query, top_k=min(top_k, 20), source_ids=source_ids)
    docs = _ask_service()._retrieve(request, tenant_id=tenant_id)

    return {
        "chunks": [
            {
                "id": d["id"],
                "content": d["document"],
                "source_id": d["metadata"].get("source_id", ""),
                "title": d["metadata"].get("title", ""),
                "chunk_index": d["metadata"].get("chunk_index"),
                "distance": round(d.get("distance", 0.0), 4),
            }
            for d in docs
        ],
        "count": len(docs),
    }


@mcp.tool()
async def index_document(
    content: str,
    source_id: str,
    title: str,
    ctx: Context,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index a text document into the knowledge base.

    Re-indexing an existing source_id fully replaces its previous content.
    Content is chunked, embedded, and stored in the vector database.

    Args:
        content:   Full text content to index (10 to 500,000 characters).
        source_id: Unique identifier for this document (slug or filename).
        title:     Human-readable title shown in source references.
        metadata:  Optional key-value metadata stored alongside each chunk.

    Returns dict with:
        source_id:      The source identifier used.
        chunks_indexed: Number of chunks stored in the index.
    """
    tenant_id = await _get_tenant_id(ctx)

    service = IndexService(
        chunker=get_chunker(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )
    request = IndexRequest(
        content=content,
        source_id=source_id,
        title=title,
        metadata=metadata or {},
    )
    response = service.index(request, tenant_id=tenant_id)

    logger.info(
        "mcp_index",
        source_id=source_id,
        chunks_indexed=response.chunks_indexed,
        tenant_id=tenant_id,
    )

    return {"source_id": response.source_id, "chunks_indexed": response.chunks_indexed}


@mcp.tool()
async def list_sources(
    ctx: Context,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """List all indexed sources in the knowledge base.

    Args:
        page:      Page number (1-based, default 1).
        page_size: Results per page (default 50, max 200).

    Returns dict with:
        sources:   List of sources (source_id, title, chunks_count, indexed_at).
        total:     Total number of sources.
        page:      Current page.
        page_size: Page size used.
    """
    tenant_id = await _get_tenant_id(ctx)
    page_size = min(page_size, 200)
    offset = (page - 1) * page_size

    raw, total = get_vector_store().list_sources(
        limit=page_size, offset=offset, tenant_id=tenant_id
    )

    return {"sources": raw, "total": total, "page": page, "page_size": page_size}


@mcp.tool()
async def delete_source(source_id: str, ctx: Context) -> dict[str, Any]:
    """Remove a source and all its chunks from the knowledge base.

    This operation is irreversible.

    Args:
        source_id: The source identifier to delete.

    Returns dict with:
        source_id:      The deleted source identifier.
        deleted_chunks: Number of chunks removed.
    """
    tenant_id = await _get_tenant_id(ctx)

    deleted = get_vector_store().delete_source(source_id, tenant_id=tenant_id)
    if deleted == 0:
        raise ValueError(f"Source '{source_id}' not found.")

    logger.info("mcp_delete_source", source_id=source_id, deleted=deleted, tenant_id=tenant_id)

    return {"source_id": source_id, "deleted_chunks": deleted}
