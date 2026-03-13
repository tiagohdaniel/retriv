from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    llm_timeout: float = 30.0

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_mode: str = "embedded"   # embedded | server
    chroma_persist_dir: str = "./chroma_data"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "documents"

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

    app_version: str = "1.0.0"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
