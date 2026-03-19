import hashlib

EMBEDDING_DIM = 768  # nomic-embed-text-v1.5 output dimensions


# Per-model task prefixes.
# nomic-embed-text-v1.5 uses "search_document:" / "search_query:"
# multilingual-e5-large uses "passage:" / "query:"
# Models absent from this dict get no prefix (raw text passed through).
_MODEL_TASK_PREFIXES: dict[str, dict[str, str]] = {
    "nomic-ai/nomic-embed-text-v1.5": {
        "document": "search_document: ",
        "query":    "search_query: ",
    },
    "intfloat/multilingual-e5-large": {
        "document": "passage: ",
        "query":    "query: ",
    },
}


class FastEmbedEmbedding:
    """Primary embedding via fastembed (ONNX Runtime, no PyTorch).

    fastembed downloads models in ONNX format from HuggingFace and handles
    tokenization, mean pooling, and L2 normalization internally.
    Designed specifically for production RAG stacks.

    Supports any fastembed model. Models that require instruction prefixes
    (nomic-embed-text-v1.5, multilingual-e5-large, etc.) are handled via
    _MODEL_TASK_PREFIXES — pass task="document" when indexing, task="query"
    when searching and the correct prefix is applied automatically.
    """

    def __init__(self, model_name: str) -> None:
        import os
        from fastembed import TextEmbedding
        kwargs = {}
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.model = TextEmbedding(model_name=model_name, **kwargs)
        self._prefixes = _MODEL_TASK_PREFIXES.get(model_name, {})

    def encode(self, texts: list[str], task: str = "document") -> list[list[float]]:
        prefix = self._prefixes.get(task, "")
        if prefix:
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
