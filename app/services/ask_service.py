from app.schemas.models import AskRequest, AskResponse, SourceReference
from app.core.logging_config import get_logger
from app.core.metrics import llm_tokens_total, ask_no_context_total
from app.core.hybrid_search import BM25Searcher, rrf_merge

logger = get_logger("retriv.ask")

_bm25 = BM25Searcher()


class AskService:
    """Orchestrates the RAG pipeline: embed → retrieve → (rerank) → prompt → LLM."""

    def __init__(
        self,
        embedding_service,
        vector_store,
        llm_client,
        observability=None,
        reranker=None,
        reranker_top_k_fetch: int = 15,
        hybrid_enabled: bool = False,
        hybrid_corpus_limit: int = 500,
    ):
        self.embedding = embedding_service
        self.vector_store = vector_store
        self.llm = llm_client
        self.observability = observability
        self.reranker = reranker
        self.reranker_top_k_fetch = reranker_top_k_fetch
        self.hybrid_enabled = hybrid_enabled
        self.hybrid_corpus_limit = hybrid_corpus_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(
        self,
        request: AskRequest,
        background_tasks=None,
        tenant_id: str | None = None,
    ) -> AskResponse:
        docs = self._retrieve(request, tenant_id)

        if not docs:
            ask_no_context_total.inc()
            logger.info("ask_no_context", question_len=len(request.question))
            return AskResponse(
                answer="No relevant documentation found. "
                       "Make sure you have indexed content via POST /index.",
            )

        prompt = self._build_prompt(request.question, docs)
        result = await self.llm.generate(prompt=prompt)
        sources = self._build_sources(docs)

        model = result.get("model", "unknown")
        tokens = result.get("tokens_used", 0)
        llm_tokens_total.labels(model=model).inc(tokens)

        logger.info(
            "ask_completed",
            docs_retrieved=len(docs),
            tokens_used=tokens,
            model=model,
            hybrid=self.hybrid_enabled,
        )

        if self.observability and background_tasks is not None:
            background_tasks.add_task(
                self.observability.trace_query,
                question=request.question,
                contexts=[doc["document"] for doc in docs],
                answer=result["answer"],
                metadata={"model": model, "docs_retrieved": len(docs), "tokens_used": tokens},
            )

        return AskResponse(
            answer=result["answer"],
            sources=sources,
            tokens_used=tokens,
            model=model,
        )

    async def ask_stream(self, request: AskRequest, tenant_id: str | None = None):
        docs = self._retrieve(request, tenant_id)

        if not docs:
            ask_no_context_total.inc()
            logger.info("ask_stream_no_context", question_len=len(request.question))
            yield {"type": "token", "content": "No relevant documentation found. Make sure you have indexed content via POST /index."}
            return

        prompt = self._build_prompt(request.question, docs)
        sources = self._build_sources(docs)

        full_answer = ""
        async for chunk in self.llm.stream(prompt=prompt):
            if isinstance(chunk, str):
                full_answer += chunk
                yield {"type": "token", "content": chunk}
            else:
                tokens = chunk.get("tokens_used", 0)
                model = chunk.get("model", "unknown")
                llm_tokens_total.labels(model=model).inc(tokens)
                logger.info(
                    "ask_stream_completed",
                    docs_retrieved=len(docs),
                    tokens_used=tokens,
                    hybrid=self.hybrid_enabled,
                )
                yield {
                    "type": "done",
                    "sources": [s.model_dump() for s in sources],
                    "tokens_used": tokens,
                    "model": model,
                }
                if self.observability:
                    await self.observability.trace_query(
                        question=request.question,
                        contexts=[doc["document"] for doc in docs],
                        answer=full_answer,
                        metadata={"model": model, "docs_retrieved": len(docs), "tokens_used": tokens},
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, request: AskRequest, tenant_id: str | None) -> list[dict]:
        """Embed → search (hybrid or semantic) → rerank → top_k docs."""
        query_embedding = self.embedding.encode([request.question])[0]
        fetch_k = self._fetch_top_k(request.top_k)

        if self.hybrid_enabled:
            try:
                corpus = self.vector_store.get_chunks(
                    tenant_id=tenant_id,
                    source_ids=request.source_ids,
                    limit=self.hybrid_corpus_limit,
                )
                bm25_docs = _bm25.search(query=request.question, corpus=corpus, top_k=fetch_k)
            except Exception as e:
                logger.warning("hybrid_bm25_fallback", error=str(e))
                bm25_docs = []
            semantic_docs = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=fetch_k,
                source_ids=request.source_ids,
                max_distance=request.max_distance,
                tenant_id=tenant_id,
            )
            docs = rrf_merge(semantic_docs, bm25_docs)[:fetch_k]
        else:
            docs = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=fetch_k,
                source_ids=request.source_ids,
                max_distance=request.max_distance,
                tenant_id=tenant_id,
            )

        if docs and self.reranker:
            docs = self.reranker.rerank(request.question, docs)[:request.top_k]

        return docs

    def _fetch_top_k(self, requested_top_k: int) -> int:
        """Expand retrieval pool when reranker is active."""
        from app.core.reranker import NullReranker
        if self.reranker and not isinstance(self.reranker, NullReranker):
            return max(self.reranker_top_k_fetch, requested_top_k)
        return requested_top_k

    def _build_prompt(self, question: str, docs: list[dict]) -> str:
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            context_blocks.append(
                f"[{i}] Source: {meta.get('title', 'unknown')}\n"
                f"{doc['document']}"
            )
        context = "\n\n".join(context_blocks)
        return (
            f"## Documentation\n\n{context}\n\n"
            f"## Question\n\n{question}"
        )

    def _build_sources(self, docs: list[dict]) -> list[SourceReference]:
        return [
            SourceReference(
                source_id=doc["metadata"].get("source_id", ""),
                title=doc["metadata"].get("title", ""),
                excerpt=doc["document"][:200] + "..." if len(doc["document"]) > 200 else doc["document"],
                relevance_score=round(doc.get("distance", 0.4), 4),
            )
            for doc in docs
        ]
