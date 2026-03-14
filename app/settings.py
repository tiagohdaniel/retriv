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
    chunk_size: int = 500
    chunk_overlap: int = 50

    # API auth
    api_auth_enabled: bool = False
    api_key: str = ""

    # Rate limiting — disabled by default for local dev
    rate_limit_enabled: bool = False
    rate_limit_index: str = "10/minute"
    rate_limit_ask: str = "30/minute"
    rate_limit_sources: str = "60/minute"

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

    # RAG evaluation — disabled by default (requires Langfuse account)
    eval_enabled: bool = False
    eval_model: str = "claude-haiku-4-5-20251001"  # lightweight model for LLM-as-judge
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    app_version: str = "1.0.0"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
