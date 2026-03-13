from app.core.agent.base import ToolBase
from app.core.vector_store import VectorStoreBase


class RagTool(ToolBase):
    """Semantic search over indexed documents.

    The foundational tool — present in every agent regardless of platform.
    Wraps the existing RAG pipeline as a pluggable tool.
    """

    name = "search_documents"
    description = (
        "Search indexed documents for information relevant to a question. "
        "Use this tool to find policies, procedures, product details, or any "
        "content that has been indexed."
    )

    def __init__(self, embedding_service, vector_store: VectorStoreBase):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def execute(self, input: dict) -> dict:
        question = input["question"]
        top_k = input.get("top_k", 5)
        max_distance = input.get("max_distance", 0.8)

        embedding = self.embedding_service.encode([question])[0]
        chunks = self.vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
            max_distance=max_distance,
        )
        return {"chunks": chunks}
