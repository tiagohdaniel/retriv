from app.schemas.models import IndexRequest, IndexResponse


class IndexService:
    """Orchestrates: chunk → embed → store.

    Re-indexing an existing source_id replaces all its chunks.
    """

    def __init__(self, chunker, embedding_service, vector_store):
        self.chunker = chunker
        self.embedding = embedding_service
        self.vector_store = vector_store

    def index(self, request: IndexRequest) -> IndexResponse:
        # delete existing chunks first — re-indexing the same source_id is idempotent
        self.vector_store.delete_source(request.source_id)

        chunks = self.chunker.chunk(request.content)
        if not chunks:
            return IndexResponse(source_id=request.source_id, chunks_indexed=0)

        # encode all chunks in a single batch call
        embeddings = self.embedding.encode(chunks)

        self.vector_store.upsert_chunks(
            source_id=request.source_id,
            title=request.title,
            chunks=chunks,
            embeddings=embeddings,
            extra_metadata=request.metadata,
        )

        return IndexResponse(source_id=request.source_id, chunks_indexed=len(chunks))
