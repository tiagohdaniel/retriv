from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    """Interface for vector store backends.

    All backends must implement these four methods.
    Services depend only on this interface — never on a concrete backend.
    Swapping backends (ChromaDB → Pinecone, pgvector, etc.) requires
    only a new implementation and a config change.

    tenant_id semantics:
        None  — no tenant filtering (auth disabled, single-tenant deploys)
        str   — filter/scope all operations to this tenant
    """

    @abstractmethod
    def upsert_chunks(
        self,
        source_id: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        tenant_id: str | None = None,
        extra_metadata: dict | None = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[str] | None = None,
        max_distance: float = 0.8,
        tenant_id: str | None = None,
    ) -> list[dict]: ...

    @abstractmethod
    def delete_source(self, source_id: str, tenant_id: str | None = None) -> int: ...

    @abstractmethod
    def list_sources(
        self,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str | None = None,
    ) -> tuple[list[dict], int]:
        """Return a page of sources and the total source count.

        ChromaDB has no native GROUP BY, so all chunk metadata is scanned
        to build the source list. Pagination is applied after aggregation,
        bounding the response payload regardless of collection size.
        True server-side pagination would require a dedicated sources table.
        """
        ...

    @abstractmethod
    def get_chunks(
        self,
        tenant_id: str | None = None,
        source_ids: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Fetch raw chunks for BM25 corpus building.

        Returns up to `limit` chunks as dicts with keys: id, document, metadata.
        Used by hybrid search — not intended for end-user responses.
        """
        ...

    @abstractmethod
    def ping(self) -> None:
        """Verifica conectividade com o backend. Lança exceção se indisponível."""
        ...
