from abc import ABC, abstractmethod


class ObservabilityBase(ABC):
    """Port for RAG observability backends.

    Implementations receive the full query context after a response is
    generated and are responsible for evaluation, tracing, and scoring.
    All calls happen in background tasks — never in the request path.
    """

    @abstractmethod
    async def trace_query(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        metadata: dict,
    ) -> None:
        """Evaluate and trace a completed RAG query."""
        ...
