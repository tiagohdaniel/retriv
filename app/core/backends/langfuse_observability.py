import json
import re

import structlog
from anthropic import AsyncAnthropic

from app.core.observability import ObservabilityBase

logger = structlog.get_logger(__name__)

_FAITHFULNESS_PROMPT = """\
You are evaluating an AI assistant's answer for faithfulness to the provided context.

Question: {question}

Context:
{context}

Answer: {answer}

Does the answer rely only on information present in the context, or does it introduce claims not supported by it?

Respond with a JSON object only, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Where 1.0 means every claim is supported by the context, 0.0 means the answer contradicts or ignores the context."""

_RELEVANCY_PROMPT = """\
You are evaluating whether an AI assistant's answer actually addresses the question asked.

Question: {question}

Answer: {answer}

Does the answer directly address the question?

Respond with a JSON object only, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Where 1.0 means the answer fully addresses the question, 0.0 means it is completely off-topic."""


class LangfuseObservability(ObservabilityBase):
    """Evaluates RAG quality via LLM-as-judge and sends traces to Langfuse.

    Evaluation uses a separate lightweight model (eval_model, default Haiku)
    to avoid the cost of scoring with the main production model.
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        anthropic_api_key: str,
        eval_model: str,
    ):
        from langfuse import Langfuse

        self._langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self._anthropic = AsyncAnthropic(api_key=anthropic_api_key)
        self._eval_model = eval_model

    async def trace_query(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        metadata: dict,
    ) -> None:
        scores = await self._evaluate(question, contexts, answer)

        trace = self._langfuse.trace(
            name="rag-query",
            input={"question": question},
            output={"answer": answer},
            metadata=metadata,
        )
        for name, value in scores.items():
            trace.score(name=name, value=value)

        self._langfuse.flush()
        logger.info("eval_trace_sent", scores=scores, **metadata)

    async def _evaluate(
        self, question: str, contexts: list[str], answer: str
    ) -> dict[str, float]:
        context_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))

        faithfulness = await self._score(
            _FAITHFULNESS_PROMPT.format(
                question=question, context=context_text, answer=answer
            )
        )
        relevancy = await self._score(
            _RELEVANCY_PROMPT.format(question=question, answer=answer)
        )

        return {"faithfulness": faithfulness, "answer_relevancy": relevancy}

    async def _score(self, prompt: str) -> float:
        try:
            message = await self._anthropic.messages.create(
                model=self._eval_model,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip() if message.content else ""
            logger.debug("eval_raw_response", text=text, stop_reason=message.stop_reason)

            # Extract JSON from response — handles leading/trailing text from the model
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in response: {text!r}")
            data = json.loads(match.group())
            return max(0.0, min(1.0, float(data["score"])))
        except Exception as exc:
            logger.warning("eval_score_failed", error=str(exc))
            return 0.0
