import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.schemas.models import IndexRequest, IndexJobResponse, JobStatusResponse
from app.dependencies import get_embedding_service, get_vector_store, get_chunker
from app.services.index_service import IndexService
from app.core.auth import verify_api_key
from app.core.rate_limit import limiter, index_rate_limit
from app.core import job_store

router = APIRouter()


@router.post("/index", response_model=IndexJobResponse, status_code=202, summary="Index a document (async)")
@limiter.limit(index_rate_limit)
async def index_document(
    request: Request,
    body: IndexRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str | None = Depends(verify_api_key),
    chunker=Depends(get_chunker),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
):
    """Start async indexing. Returns a job_id immediately.

    Poll GET /index/status/{job_id} to check progress.
    Status transitions: processing → done | failed.
    """
    job_id = job_store.create(body.source_id)
    background_tasks.add_task(
        _run_index_background, job_id, body, tenant_id, chunker, embedding_service, vector_store
    )
    return IndexJobResponse(job_id=job_id, source_id=body.source_id)


@router.get("/index/status/{job_id}", response_model=JobStatusResponse, summary="Check indexing job status")
def index_status(
    job_id: str,
    _: str | None = Depends(verify_api_key),
):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job_id, **job)


# ── background task ───────────────────────────────────────────────────────────

async def _run_index_background(job_id: str, body: IndexRequest, tenant_id, chunker, embedding_service, vector_store):
    """Wraps the CPU-bound indexing in a thread so the event loop stays free for status polls."""
    await asyncio.to_thread(_run_index, job_id, body, tenant_id, chunker, embedding_service, vector_store)


def _run_index(job_id: str, body: IndexRequest, tenant_id, chunker, embedding_service, vector_store):
    try:
        service = IndexService(
            chunker=chunker,
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        result = service.index(body, tenant_id=tenant_id)
        job_store.complete(job_id, result.chunks_indexed)
    except Exception as e:
        job_store.fail(job_id, str(e))
