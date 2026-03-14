import json

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse

from app.schemas.models import AskRequest, AskResponse
from app.dependencies import get_embedding_service, get_vector_store, get_llm_client, get_observability
from app.services.ask_service import AskService
from app.core.auth import verify_api_key
from app.core.rate_limit import limiter, ask_rate_limit

router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="Query indexed documents")
@limiter.limit(ask_rate_limit)
async def ask_question(
    request: Request,
    body: AskRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str | None = Depends(verify_api_key),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    llm_client=Depends(get_llm_client),
    observability=Depends(get_observability),
):
    """Ask a natural language question over indexed documents.

    Returns an LLM-generated answer grounded on retrieved chunks,
    with source references and relevance scores.
    """
    service = AskService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_client=llm_client,
        observability=observability,
    )
    return await service.ask(body, background_tasks=background_tasks, tenant_id=tenant_id)


@router.post("/ask/stream", summary="Query indexed documents with streaming")
@limiter.limit(ask_rate_limit)
async def ask_question_stream(
    request: Request,
    body: AskRequest,
    tenant_id: str | None = Depends(verify_api_key),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    llm_client=Depends(get_llm_client),
):
    service = AskService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_client=llm_client,
    )

    async def event_stream():
        async for chunk in service.ask_stream(body, tenant_id=tenant_id):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
