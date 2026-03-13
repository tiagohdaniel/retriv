"""Custom Prometheus metrics for the retriv RAG pipeline.

Standard HTTP metrics (request count, duration, status codes) are handled
automatically by prometheus-fastapi-instrumentator.

These counters cover domain-specific signals:
- LLM token consumption — cost visibility
- Indexing throughput — operational health
- No-context responses — retrieval quality signal
"""
from prometheus_client import Counter

llm_tokens_total = Counter(
    "retriv_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model"],
)

documents_indexed_total = Counter(
    "retriv_documents_indexed_total",
    "Total documents successfully indexed",
)

chunks_indexed_total = Counter(
    "retriv_chunks_indexed_total",
    "Total chunks stored in the vector store",
)

ask_no_context_total = Counter(
    "retriv_ask_no_context_total",
    "Queries that returned no relevant context after distance filtering",
)
