from fastapi import APIRouter, Depends, Request

from app.schemas.models import IndexRequest, IndexResponse
from app.dependencies import get_embedding_service, get_vector_store, get_chunker
from app.services.index_service import IndexService
from app.core.auth import verify_api_key
from app.core.rate_limit import limiter, index_rate_limit

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/index", response_model=IndexResponse, summary="Index a document")
@limiter.limit(index_rate_limit)
def index_document(
    request: Request,
    body: IndexRequest,
    chunker=Depends(get_chunker),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
):
    """Index a document for semantic search. Re-indexing replaces existing chunks."""
    service = IndexService(
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    return service.index(body)
