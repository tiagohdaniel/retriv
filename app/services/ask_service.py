from app.schemas.models import AskRequest, AskResponse, SourceReference


class AskService:
    """Orchestrates the RAG pipeline: embed → search → prompt → LLM → response."""

    def __init__(self, embedding_service, vector_store, llm_client):
        self.embedding = embedding_service
        self.vector_store = vector_store
        self.llm = llm_client

    async def ask(self, request: AskRequest) -> AskResponse:
        query_embedding = self.embedding.encode([request.question])[0]

        docs = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            source_ids=request.source_ids,
            max_distance=request.max_distance,
        )

        # skip LLM call if no relevant context — avoids hallucination and saves tokens
        if not docs:
            return AskResponse(
                answer="No relevant documentation found. "
                       "Make sure you have indexed content via POST /index.",
            )

        prompt = self._build_prompt(request.question, docs)
        result = await self.llm.generate(prompt=prompt)
        sources = self._build_sources(docs)

        return AskResponse(
            answer=result["answer"],
            sources=sources,
            tokens_used=result.get("tokens_used", 0),
            model=result.get("model", ""),
        )

    async def ask_stream(self, request: AskRequest):
        query_embedding = self.embedding.encode([request.question])[0]

        docs = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            source_ids=request.source_ids,
            max_distance=request.max_distance,
        )

        if not docs:
            yield {"type": "token", "content": "No relevant documentation found. Make sure you have indexed content via POST /index."}
            return

        prompt = self._build_prompt(request.question, docs)
        sources = self._build_sources(docs)

        async for chunk in self.llm.stream(prompt=prompt):
            if isinstance(chunk, str):
                yield {"type": "token", "content": chunk}
            else:
                yield {
                    "type": "done",
                    "sources": [s.model_dump() for s in sources],
                    "tokens_used": chunk.get("tokens_used", 0),
                    "model": chunk.get("model", ""),
                }

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
                relevance_score=round(doc["distance"], 4),
            )
            for doc in docs
        ]
