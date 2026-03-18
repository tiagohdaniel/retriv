from app.schemas.models import AskRequest, AskResponse, SourceReference
from app.core.logging_config import get_logger
from app.core.metrics import llm_tokens_total, ask_no_context_total
from app.core.hybrid_search import BM25Searcher, rrf_merge
from app.core.query_normalizer import QueryNormalizer

logger = get_logger("retriv.ask")

_bm25 = BM25Searcher()
_normalizer = QueryNormalizer()


def _source_distribution(docs: list[dict]) -> dict[str, int]:
    """Return {source_id: chunk_count} for the retrieved docs."""
    counts: dict[str, int] = {}
    for doc in docs:
        source_id = doc.get("metadata", {}).get("source_id", "unknown")
        counts[source_id] = counts.get(source_id, 0) + 1
    return counts


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
        source_diversity_enabled: bool = False,
        source_diversity_max_per_source: int = 3,
        neighbor_expansion_enabled: bool = False,
    ):
        self.embedding = embedding_service
        self.vector_store = vector_store
        self.llm = llm_client
        self.observability = observability
        self.reranker = reranker
        self.reranker_top_k_fetch = reranker_top_k_fetch
        self.hybrid_enabled = hybrid_enabled
        self.hybrid_corpus_limit = hybrid_corpus_limit
        self.source_diversity_enabled = source_diversity_enabled
        self.source_diversity_max_per_source = source_diversity_max_per_source
        self.neighbor_expansion_enabled = neighbor_expansion_enabled

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

        prompt = self._build_prompt(request.question, docs, request.history)
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
                metadata={
                    "model": model,
                    "docs_retrieved": len(docs),
                    "tokens_used": tokens,
                    "hybrid": self.hybrid_enabled,
                    "source_distribution": _source_distribution(docs),
                },
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

        prompt = self._build_prompt(request.question, docs, request.history)
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
                        metadata={
                            "model": model,
                            "docs_retrieved": len(docs),
                            "tokens_used": tokens,
                            "hybrid": self.hybrid_enabled,
                            "source_distribution": _source_distribution(docs),
                        },
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, request: AskRequest, tenant_id: str | None) -> list[dict]:
        """Embed → search (hybrid or semantic) → diversity → rerank → top_k docs."""
        retrieval_query, was_normalized = _normalizer.normalize(request.question)
        if was_normalized:
            logger.debug(
                "query_normalized",
                original=request.question,
                normalized=retrieval_query,
            )

        query_embedding = self.embedding.encode([retrieval_query], task="query")[0]
        fetch_k = self._fetch_top_k(request.top_k)

        if self.hybrid_enabled:
            try:
                corpus = self.vector_store.get_chunks(
                    tenant_id=tenant_id,
                    source_ids=request.source_ids,
                    limit=self.hybrid_corpus_limit,
                )
                bm25_docs = _bm25.search(query=retrieval_query, corpus=corpus, top_k=fetch_k)
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

        if self.source_diversity_enabled and docs:
            pre_diversity = len(docs)
            docs = self._apply_source_diversity(docs, self.source_diversity_max_per_source)
            logger.debug(
                "source_diversity_applied",
                pre=pre_diversity,
                post=len(docs),
                max_per_source=self.source_diversity_max_per_source,
            )

        # Expand neighbors before reranking so the cross-encoder evaluates
        # adjacent chunks alongside candidates — not after the ranking decision.
        if self.neighbor_expansion_enabled and docs:
            pre_expansion = len(docs)
            docs = self._expand_neighbors(docs, tenant_id)
            logger.debug(
                "neighbor_expansion_applied",
                pre=pre_expansion,
                post=len(docs),
            )

        if docs and self.reranker:
            logger.debug("rerank_input", count=len(docs))
            docs = self.reranker.rerank(request.question, docs)[:request.top_k]
            # Re-sort top-k by document position for coherent LLM context window.
            docs.sort(key=lambda d: (
                d.get("metadata", {}).get("source_id", ""),
                d.get("metadata", {}).get("chunk_index", 0),
            ))
            logger.debug("rerank_output", count=len(docs))

        return docs

    def _fetch_top_k(self, requested_top_k: int) -> int:
        """Expand retrieval pool when reranker is active."""
        from app.core.reranker import NullReranker
        if self.reranker and not isinstance(self.reranker, NullReranker):
            return max(self.reranker_top_k_fetch, requested_top_k)
        return requested_top_k

    def _expand_neighbors(self, docs: list[dict], tenant_id: str | None) -> list[dict]:
        """Add chunk_index ± 1 from the same source_id for each retrieved doc.

        Uses chunk_index and source_id from metadata to construct neighbor IDs.
        Neighbors that do not exist in the store are silently skipped.
        The result is sorted by (source_id, chunk_index) so adjacent chunks
        appear together in the LLM context window.
        """
        existing_ids: set[str] = {doc["id"] for doc in docs}
        neighbor_ids: list[str] = []

        for doc in docs:
            meta = doc.get("metadata", {})
            source_id = meta.get("source_id")
            chunk_index = meta.get("chunk_index")
            if source_id is None or chunk_index is None:
                continue
            prefix = f"{tenant_id}__{source_id}" if tenant_id else source_id
            for delta in (-1, 1):
                nid = f"{prefix}__chunk_{chunk_index + delta}"
                if nid not in existing_ids:
                    neighbor_ids.append(nid)
                    existing_ids.add(nid)

        neighbors = self.vector_store.fetch_by_ids(neighbor_ids)
        all_docs = docs + neighbors
        all_docs.sort(key=lambda d: (
            d.get("metadata", {}).get("source_id", ""),
            d.get("metadata", {}).get("chunk_index", 0),
        ))
        return all_docs

    def _apply_source_diversity(self, docs: list[dict], max_per_source: int) -> list[dict]:
        """Cap how many chunks from the same source_id appear in the pool.

        Preserves ranking order — first N chunks per source are kept, the rest
        are dropped. This prevents a single large document from dominating the
        retrieval pool and crowding out relevant context from other sources.
        """
        counts: dict[str, int] = {}
        result = []
        for doc in docs:
            source_id = doc.get("metadata", {}).get("source_id", "")
            count = counts.get(source_id, 0)
            if count < max_per_source:
                result.append(doc)
                counts[source_id] = count + 1
        return result

    def _build_prompt(self, question: str, docs: list[dict], history: list | None = None) -> str:
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            context_blocks.append(
                f"[{i}] Source: {meta.get('title', 'unknown')}\n"
                f"{doc['document']}"
            )
        context = "\n\n".join(context_blocks)

        history_section = ""
        if history:
            lines = []
            for msg in history:
                prefix = "User" if msg.role == "user" else "Assistant"
                lines.append(f"{prefix}: {msg.content}")
            history_section = "## Previous conversation\n\n" + "\n\n".join(lines) + "\n\n"

        return (
            f"## Documentation\n\n{context}\n\n"
            f"{history_section}"
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
