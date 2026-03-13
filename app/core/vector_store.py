from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    """Interface for vector store backends.

    All backends must implement these four methods.
    Services depend only on this interface — never on a concrete backend.
    Swapping backends (ChromaDB → Pinecone, pgvector, etc.) requires
    only a new implementation and a config change.
    """

    @abstractmethod
    def upsert_chunks(
        self,
        source_id: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        extra_metadata: dict | None = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[str] | None = None,
        max_distance: float = 0.8,
    ) -> list[dict]: ...

    @abstractmethod
    def delete_source(self, source_id: str) -> int: ...

    @abstractmethod
    def list_sources(self) -> list[dict]: ...
