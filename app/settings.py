from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-4-20250514"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "documents"
    chunk_size: int = 500
    chunk_overlap: int = 50
    app_version: str = "1.0.0"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
