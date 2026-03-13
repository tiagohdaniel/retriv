from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.schemas.models import SourcesResponse, SourceItem
from app.dependencies import get_vector_store
from app.core.auth import verify_api_key
from app.core.rate_limit import limiter, sources_rate_limit

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/sources", response_model=SourcesResponse, summary="List indexed sources")
@limiter.limit(sources_rate_limit)
def list_sources(
    request: Request,
    vector_store=Depends(get_vector_store),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=50, ge=1, le=200, description="Results per page"),
):
    """List indexed sources with pagination."""
    offset = (page - 1) * page_size
    raw, total = vector_store.list_sources(limit=page_size, offset=offset)
    sources = [SourceItem(**s) for s in raw]
    return SourcesResponse(sources=sources, total=total, page=page, page_size=page_size)


@router.delete("/sources/{source_id}", summary="Delete an indexed source")
@limiter.limit(sources_rate_limit)
def delete_source(source_id: str, request: Request, vector_store=Depends(get_vector_store)):
    """Remove all chunks for a source from the index."""
    deleted = vector_store.delete_source(source_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")
    return {"deleted_chunks": deleted, "source_id": source_id}
