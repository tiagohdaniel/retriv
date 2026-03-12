import hashlib

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimensions


class SentenceTransformerEmbedding:
    """Level 1 — full sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()


class ONNXEmbedding:
    """Level 2 — ONNX Runtime via ChromaDB (same model, lighter runtime).

    Downloads all-MiniLM-L6-v2 in ONNX format (~87MB) on first use.
    Cached at ~/.cache/chroma/onnx_models/
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = self._ef(list(texts))
        return [r.tolist() if hasattr(r, "tolist") else list(r) for r in results]


class HashEmbedding:
    """Level 3 — deterministic fallback. No semantic similarity.

    Used in tests and environments where no model is available.
    Vectors are reproducible but NOT semantically meaningful.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        extended = (h * (EMBEDDING_DIM // len(h) + 1))[:EMBEDDING_DIM]
        return [(b / 127.5 - 1.0) for b in extended]


def create_embedding_service(model_name: str = "all-MiniLM-L6-v2"):
    """Factory with 3-level fallback.

    SentenceTransformer → ONNX → HashEmbedding
    """
    try:
        return SentenceTransformerEmbedding(model_name)
    except Exception:
        pass
    try:
        return ONNXEmbedding()
    except Exception:
        pass
    return HashEmbedding()
