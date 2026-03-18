import hashlib

EMBEDDING_DIM = 768  # nomic-embed-text-v1.5 output dimensions


_TASK_PREFIXES = {
    "document": "search_document: ",
    "query": "search_query: ",
}

_PREFIX_MODELS = {"nomic-ai/nomic-embed-text-v1.5"}


class FastEmbedEmbedding:
    """Primary embedding via fastembed (ONNX Runtime, no PyTorch).

    fastembed downloads models in ONNX format from HuggingFace and handles
    tokenization, mean pooling, and L2 normalization internally.
    Designed specifically for production RAG stacks.

    Supports nomic-embed-text-v1.5 natively. The model uses instruction
    prefixes for best quality:
      - documents: "search_document: " + text
      - queries:   "search_query: " + text
    Pass task="document" when indexing, task="query" when searching.
    """

    def __init__(self, model_name: str) -> None:
        import os
        from fastembed import TextEmbedding
        kwargs = {}
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.model = TextEmbedding(model_name=model_name, **kwargs)
        self._use_prefixes = model_name in _PREFIX_MODELS

    def encode(self, texts: list[str], task: str = "document") -> list[list[float]]:
        if self._use_prefixes:
            prefix = _TASK_PREFIXES.get(task, "")
            texts = [prefix + t for t in texts]
        return [emb.tolist() for emb in self.model.embed(texts)]


class ONNXEmbedding:
    """ONNX Runtime via ChromaDB built-in — all-MiniLM-L6-v2 only.

    Kept as fallback for MiniLM-family models when fastembed is unavailable.
    Downloads ~87MB on first use. Not used for other models: a different model
    in ONNX would produce vectors of a different dimension, silently breaking
    stored embeddings.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts: list[str], task: str = "document") -> list[list[float]]:
        results = self._ef(list(texts))
        return [r.tolist() if hasattr(r, "tolist") else list(r) for r in results]


class HashEmbedding:
    """Deterministic fallback — no semantic similarity.

    Used in tests and environments where no model is available.
    Vectors are reproducible but NOT semantically meaningful.
    """

    def encode(self, texts: list[str], task: str = "document") -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        extended = (h * (EMBEDDING_DIM // len(h) + 1))[:EMBEDDING_DIM]
        return [(b / 127.5 - 1.0) for b in extended]


def create_embedding_service(model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
    """Factory with 3-level fallback.

    FastEmbed/ONNX (preferred) → ONNX MiniLM (MiniLM models only) → HashEmbedding

    The MiniLM ONNX fallback only activates for MiniLM-family models to avoid
    silently returning wrong-dimension vectors for other models.
    """
    try:
        return FastEmbedEmbedding(model_name)
    except Exception:
        pass

    if "minilm" in model_name.lower():
        try:
            return ONNXEmbedding()
        except Exception:
            pass

    return HashEmbedding()
