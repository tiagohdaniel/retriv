from abc import ABC, abstractmethod


class RerankerBase(ABC):
    """Interface for cross-encoder rerankers.

    Receives the search query and a list of candidate docs (as returned by
    VectorStoreBase.search), and returns them re-sorted by relevance score.
    """

    @abstractmethod
    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """Re-order docs by relevance to query. Returns same dicts, re-sorted."""
        ...


class NullReranker(RerankerBase):
    """No-op — returns docs unchanged. Used when RERANKER_ENABLED=false."""

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        return docs


class FastEmbedReranker(RerankerBase):
    """Cross-encoder reranker via fastembed TextCrossEncoder.

    Downloads the ONNX model on first instantiation (same pattern as embeddings).
    Scores each (query, document) pair independently — more accurate than
    cosine similarity because the cross-encoder sees both texts together.

    Recommended model: BAAI/bge-reranker-base
    - Multilingual (handles Portuguese)
    - ~280 MB ONNX
    - Strong quality/speed trade-off for production RAG

    Lighter alternative: Xenova/ms-marco-MiniLM-L-6-v2 (~22 MB, English-only)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        if not docs:
            return docs
        passages = [doc["document"] for doc in docs]
        scores = list(self.model.rerank(query, passages))
        # Sort descending by score — higher = more relevant
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]
