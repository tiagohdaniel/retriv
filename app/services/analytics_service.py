"""Analytics service — structured data analysis pipeline.

Pipeline:
    base64 decode → DataAnalyzer (pandas) → prompt → LLM → answer

The analyzer performs deterministic aggregation; the LLM only formats
the pre-computed result into natural language. This ensures correctness
(pandas sums accurately, LLMs don't) and bounds token usage to O(1).
"""
import base64
import json

from app.core.data_analyzer import DataAnalyzerBase
from app.core.llm_client import LLMClientBase
from app.core.logging_config import get_logger
from app.core.metrics import analytics_requests_total, analytics_tokens_total, analytics_errors_total
from app.schemas.models import AnalyzeRequest, AnalyzeResponse

logger = get_logger("retriv.analytics")


class AnalyticsService:
    def __init__(self, analyzer: DataAnalyzerBase, llm_client: LLMClientBase) -> None:
        self.analyzer = analyzer
        self.llm = llm_client

    async def analyze(self, request: AnalyzeRequest, tenant_id: str | None = None) -> AnalyzeResponse:
        analytics_requests_total.inc()

        try:
            file_content = base64.b64decode(request.source.content)
        except Exception:
            analytics_errors_total.labels(reason="decode_error").inc()
            raise ValueError("Invalid base64 content in 'source.content'.")

        try:
            analysis = self.analyzer.analyze(
                question=request.question,
                file_content=file_content,
                file_type=request.source.type.lower(),
            )
        except ValueError as exc:
            analytics_errors_total.labels(reason="analysis_error").inc()
            raise

        prompt = self._build_prompt(request.question, analysis)
        llm_result = await self.llm.generate(prompt=prompt)

        model = llm_result.get("model", "unknown")
        tokens = llm_result.get("tokens_used", 0)
        analytics_tokens_total.labels(model=model).inc(tokens)

        logger.info(
            "analytics_completed",
            op_summary=analysis.summary,
            row_count=analysis.row_count,
            tokens_used=tokens,
            model=model,
            tenant_id=tenant_id,
        )

        computation = {
            "summary": analysis.summary,
            "result": analysis.result,
            "row_count": analysis.row_count,
            "columns": analysis.columns,
        }

        return AnalyzeResponse(
            answer=llm_result["answer"],
            computation=computation,
            tokens_used=tokens,
            model=model,
        )

    def _build_prompt(self, question: str, analysis) -> str:
        result_text = json.dumps(analysis.result, ensure_ascii=False, indent=2, default=str)
        columns_text = ", ".join(analysis.columns[:20])  # cap to avoid very long prompts
        if len(analysis.columns) > 20:
            columns_text += f" ... (+{len(analysis.columns) - 20} more)"

        return (
            f"## Dataset Information\n\n"
            f"Columns: {columns_text}\n"
            f"Total rows: {analysis.row_count:,}\n\n"
            f"## Computed Result\n\n"
            f"Operation: {analysis.summary}\n\n"
            f"```json\n{result_text}\n```\n\n"
            f"## Question\n\n"
            f"{question}\n\n"
            f"## Instructions\n\n"
            f"Answer the question based strictly on the computed result above. "
            f"Be concise and precise. Use the same language as the question. "
            f"Format numbers clearly (use thousands separators when appropriate). "
            f"Do not invent values not present in the result."
        )
