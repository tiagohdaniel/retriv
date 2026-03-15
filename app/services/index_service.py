from app.schemas.models import IndexRequest, IndexResponse
from app.core.logging_config import get_logger
from app.core.metrics import documents_indexed_total, chunks_indexed_total

logger = get_logger("retriv.index")

_EMBED_BATCH_SIZE = 32  # chunks per ONNX inference call — prevents OOM on large documents


class IndexService:
    """Orchestrates: chunk → embed → store.

    Re-indexing an existing source_id replaces all its chunks.
    """

    def __init__(self, chunker, embedding_service, vector_store):
        self.chunker = chunker
        self.embedding = embedding_service
        self.vector_store = vector_store

    def index(self, request: IndexRequest, tenant_id: str | None = None) -> IndexResponse:
        # delete existing chunks first — re-indexing the same source_id is idempotent
        self.vector_store.delete_source(request.source_id, tenant_id=tenant_id)

        chunks = self.chunker.chunk(request.content)
        if not chunks:
            logger.warning("index_no_chunks", source_id=request.source_id)
            return IndexResponse(source_id=request.source_id, chunks_indexed=0)

        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
            embeddings.extend(self.embedding.encode(chunks[i : i + _EMBED_BATCH_SIZE]))

        self.vector_store.upsert_chunks(
            source_id=request.source_id,
            title=request.title,
            chunks=chunks,
            embeddings=embeddings,
            tenant_id=tenant_id,
            extra_metadata=request.metadata,
        )

        documents_indexed_total.inc()
        chunks_indexed_total.inc(len(chunks))
        logger.info(
            "index_completed",
            source_id=request.source_id,
            chunks_indexed=len(chunks),
            tenant_id=tenant_id,
        )
        return IndexResponse(source_id=request.source_id, chunks_indexed=len(chunks))
