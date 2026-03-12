from datetime import datetime, timezone


class DocsVectorStore:
    """ChromaDB-backed vector store for document chunks.

    Each chunk is stored with its source metadata, enabling
    filtered search and full source management (list, delete).
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
        extra_metadata: dict | None = None,
    ) -> None:
        indexed_at = datetime.now(timezone.utc).isoformat()
        ids = [f"{source_id}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_id": source_id,
                "title": title,
                "chunk_index": i,
                "indexed_at": indexed_at,
                **(extra_metadata or {}),
            }
            for i in range(len(chunks))
        ]
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
    ) -> list[dict]:
        """Return chunks within max_distance (cosine). 0 = identical, 1 = opposite."""
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        where = self._build_where(source_ids)

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

    def delete_source(self, source_id: str) -> int:
        result = self.collection.get(
            where={"source_id": source_id},
            include=[],
        )
        ids = result["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    # TODO: pagination — this loads all metadata into memory, will be slow at scale
    def list_sources(self) -> list[dict]:
        if self.collection.count() == 0:
            return []

        result = self.collection.get(include=["metadatas"])
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

        return list(sources.values())

    def _build_where(self, source_ids: list[str] | None) -> dict | None:
        if not source_ids:
            return None
        if len(source_ids) == 1:
            return {"source_id": source_ids[0]}
        return {"$or": [{"source_id": sid} for sid in source_ids]}
