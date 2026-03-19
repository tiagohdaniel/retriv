from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    source_id: str = Field(max_length=200, description="Unique identifier for this document")
    title: str = Field(max_length=500, description="Human-readable title")
    content: str = Field(min_length=10, max_length=500_000, description="Full text content to index")
    metadata: dict = Field(default_factory=dict, description="Optional extra metadata")


class IndexResponse(BaseModel):
    status: str = "ok"
    source_id: str
    chunks_indexed: int


class IndexJobResponse(BaseModel):
    job_id: str
    status: str = "processing"
    source_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # processing | done | failed
    source_id: str
    chunks_indexed: int = 0
    error: str | None = None


class ConversationMessage(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(max_length=4_000)


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1_000, description="Natural language question")
    top_k: int = Field(default=10, ge=1, le=20, description="Number of chunks to retrieve")
    source_ids: list[str] | None = Field(
        default=None, description="Restrict search to specific sources"
    )
    max_distance: float = Field(
        default=0.6, ge=0.0, le=2.0,
        description="Maximum cosine distance (0=identical, 2=opposite). Chunks above this threshold are discarded."
    )
    history: list[ConversationMessage] | None = Field(
        default=None,
        description="Previous conversation turns (up to 6 messages). Enables follow-up questions.",
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
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    checks: dict[str, str] = Field(default_factory=dict)


class DataSource(BaseModel):
    type: str = Field(description="File type: 'csv', 'xlsx', or 'json'")
    content: str = Field(description="Base64-encoded file content")
    filename: str | None = Field(default=None, description="Original filename (optional, used as context)")


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1_000, description="Natural language question about the data")
    source: DataSource


class AnalyzeResponse(BaseModel):
    answer: str
    computation: dict | None = Field(default=None, description="Raw computation result for client-side use")
    tokens_used: int = 0
    model: str = ""
