import base64

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.models import AnalyzeRequest, AnalyzeResponse
from app.dependencies import get_analyzer, get_llm_client, get_settings
from app.services.analytics_service import AnalyticsService
from app.core.auth import verify_api_key
from app.core.rate_limit import limiter, analyze_rate_limit

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze structured data")
@limiter.limit(analyze_rate_limit)
async def analyze_data(
    request: Request,
    body: AnalyzeRequest,
    tenant_id: str | None = Depends(verify_api_key),
    analyzer=Depends(get_analyzer),
    llm_client=Depends(get_llm_client),
    settings=Depends(get_settings),
):
    """Analyze structured data (CSV, XLSX, JSON) using a natural language question.

    The file must be base64-encoded in `source.content`.
    Computation is deterministic (pandas) — the LLM only formats the result.

    Supported operations (inferred from the question):
    - Aggregation: sum/total, mean/média, max/maior, min/menor
    - Count: quantos, count
    - Listing: listar, top N
    - Descriptive stats: default when no operation keyword is found
    """
    max_bytes = settings.analytics_max_file_size_mb * 1024 * 1024
    # base64 encodes 3 binary bytes into 4 chars — decoded size ≈ len(b64) * 3/4
    if len(body.source.content) * 3 // 4 > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.analytics_max_file_size_mb} MB.",
        )

    service = AnalyticsService(analyzer=analyzer, llm_client=llm_client)

    try:
        return await service.analyze(body, tenant_id=tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
