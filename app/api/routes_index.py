from fastapi import APIRouter, Depends

from app.schemas.models import IndexRequest, IndexResponse
from app.dependencies import get_embedding_service, get_vector_store, get_chunker
from app.services.index_service import IndexService

router = APIRouter()


@router.post("/index", response_model=IndexResponse, summary="Index a document")
def index_document(
    request: IndexRequest,
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
    return service.index(request)
