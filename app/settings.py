from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_backend: str = "anthropic"  # anthropic | openai (future)
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    llm_timeout: float = 30.0

    # Embeddings
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # Vector store
    chroma_mode: str = "embedded"   # embedded | server | cloud
    chroma_persist_dir: str = "./chroma_data"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "documents"
    chroma_cloud_api_key: str = ""
    chroma_cloud_tenant: str = ""
    chroma_cloud_database: str = ""

    # Chunking
    chunking_strategy: str = "semantic"  # "semantic" | "fixed"
    chunk_size: int = 800               # chars; semantic works better with larger values
    chunk_overlap: int = 100

    # API auth
    api_auth_enabled: bool = False
    api_key: str = ""
    # Multi-tenant: comma-separated "key:tenant_id" pairs
    # Example: "abc123:guerreiro,xyz789:porto"
    # When set, api_key (single-tenant) is ignored.
    api_keys: str = ""

    # Rate limiting — disabled by default for local dev
    rate_limit_enabled: bool = False
    rate_limit_index: str = "10/minute"
    rate_limit_ask: str = "30/minute"
    rate_limit_sources: str = "60/minute"
    rate_limit_analyze: str = "20/minute"

    # Analytics pipeline
    analytics_max_rows: int = 100_000
    analytics_max_file_size_mb: int = 10

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"        # "json" | "console"

    # HTTP
    cors_origins: str = "*"         # comma-separated list or "*"
    web_concurrency: int = 2

    # Observability
    metrics_enabled: bool = True    # exposes GET /metrics (Prometheus format)
    metrics_username: str = ""      # Basic auth for /metrics — leave empty to disable auth
    metrics_password: str = ""

    # Hybrid search (BM25 + semantic → RRF) — disabled by default
    hybrid_enabled: bool = False
    hybrid_bm25_corpus_limit: int = 200  # max chunks loaded into BM25 corpus (Chroma Cloud limit: 300)

    # Reranker — disabled by default (requires model download on first use)
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"  # multilingual; alt: Xenova/ms-marco-MiniLM-L-6-v2
    reranker_top_k_fetch: int = 15  # how many candidates to retrieve before reranking
    reranker_top_n: int = 5         # how many to keep after reranking (sent to LLM)

    # Source diversity — cap chunks per source in retrieval pool (prevents one doc dominating)
    source_diversity_enabled: bool = False
    source_diversity_max_per_source: int = 3  # max chunks from the same source_id in pool

    # Neighbor expansion — include adjacent chunks (index ± 1) from the same source
    # Prevents split sentences/sections from landing in separate LLM context windows
    neighbor_expansion_enabled: bool = False

    # RAG evaluation — disabled by default (requires Langfuse account)
    eval_enabled: bool = False
    eval_model: str = "claude-haiku-4-5-20251001"  # lightweight model for LLM-as-judge
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    app_version: str = "1.2.0"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
