from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    source_id: str = Field(description="Unique identifier for this document")
    title: str = Field(description="Human-readable title")
    content: str = Field(min_length=10, description="Full text content to index")
    metadata: dict = Field(default_factory=dict, description="Optional extra metadata")


class IndexResponse(BaseModel):
    status: str = "ok"
    source_id: str
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str = Field(min_length=5, description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    source_ids: list[str] | None = Field(
        default=None, description="Restrict search to specific sources"
    )
    max_distance: float = Field(
        default=0.8, ge=0.0, le=2.0,
        description="Maximum cosine distance (0=identical, 2=opposite). Chunks above this threshold are discarded."
    )


class SourceReference(BaseModel):
    source_id: str
    title: str
    excerpt: str
    relevance_score: float = Field(description="Cosine similarity score (0=identical, 1=opposite)")


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    tokens_used: int = 0
    model: str = ""


class SourceItem(BaseModel):
    source_id: str
    title: str
    chunks_count: int
    indexed_at: str


class SourcesResponse(BaseModel):
    sources: list[SourceItem]
    total: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
