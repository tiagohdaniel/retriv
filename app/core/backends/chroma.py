from datetime import datetime, timezone

from app.core.vector_store import VectorStoreBase


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB implementation of VectorStoreBase.

    Supports both embedded (local dev) and server mode (production).
    The client is injected — use dependencies.py to build the right one.
    """

    def __init__(self, chroma_client, collection_name: str = "documents"):
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        source_id: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        tenant_id: str | None = None,
        extra_metadata: dict | None = None,
    ) -> None:
        indexed_at = datetime.now(timezone.utc).isoformat()
        # Prefix IDs with tenant to avoid collisions across tenants
        prefix = f"{tenant_id}__{source_id}" if tenant_id else source_id
        ids = [f"{prefix}__chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        for i in range(len(chunks)):
            meta: dict = {
                "source_id": source_id,
                "title": title,
                "chunk_index": i,
                "indexed_at": indexed_at,
                **(extra_metadata or {}),
            }
            if tenant_id is not None:
                meta["tenant_id"] = tenant_id
            metadatas.append(meta)
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[str] | None = None,
        max_distance: float = 0.8,
        tenant_id: str | None = None,
    ) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        where = self._build_where(source_ids=source_ids, tenant_id=tenant_id)

        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            if distance > max_distance:
                continue
            docs.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                }
            )
        return docs

    def delete_source(self, source_id: str, tenant_id: str | None = None) -> int:
        where: dict
        if tenant_id is not None:
            where = {"$and": [{"source_id": source_id}, {"tenant_id": tenant_id}]}
        else:
            where = {"source_id": source_id}
        result = self.collection.get(where=where, include=[])
        ids = result["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def list_sources(
        self,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str | None = None,
    ) -> tuple[list[dict], int]:
        if self.collection.count() == 0:
            return [], 0

        kwargs: dict = {"include": ["metadatas"]}
        if tenant_id is not None:
            kwargs["where"] = {"tenant_id": tenant_id}

        result = self.collection.get(**kwargs)
        sources: dict[str, dict] = {}

        for meta in result["metadatas"]:
            sid = meta["source_id"]
            if sid not in sources:
                sources[sid] = {
                    "source_id": sid,
                    "title": meta.get("title", sid),
                    "chunks_count": 0,
                    "indexed_at": meta.get("indexed_at", ""),
                }
            sources[sid]["chunks_count"] += 1

        all_sources = list(sources.values())
        total = len(all_sources)
        page = all_sources[offset: offset + limit]
        return page, total

    def get_chunks(
        self,
        tenant_id: str | None = None,
        source_ids: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict]:
        where = self._build_where(source_ids=source_ids, tenant_id=tenant_id)
        kwargs: dict = {"include": ["documents", "metadatas"], "limit": limit}
        if where:
            kwargs["where"] = where
        result = self.collection.get(**kwargs)
        return [
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta,
            }
            for doc_id, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def fetch_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        result = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        return [
            {"id": doc_id, "document": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def ping(self) -> None:
        self.collection.count()

    def _build_where(
        self,
        source_ids: list[str] | None,
        tenant_id: str | None,
    ) -> dict | None:
        conditions: list[dict] = []

        if tenant_id is not None:
            conditions.append({"tenant_id": tenant_id})

        if source_ids:
            if len(source_ids) == 1:
                conditions.append({"source_id": source_ids[0]})
            else:
                conditions.append({"$or": [{"source_id": sid} for sid in source_ids]})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
