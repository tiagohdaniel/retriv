from functools import lru_cache

import chromadb

from app.settings import Settings
from app.core.chunker import TextChunker
from app.core.embeddings import create_embedding_service
from app.core.vector_store import VectorStoreBase
from app.core.llm_client import LLMClientBase
from app.core.observability import ObservabilityBase
from app.core.backends.chroma import ChromaVectorStore
from app.core.backends.anthropic import AnthropicClient
from app.core.backends.null_observability import NullObservability
from app.core.backends.langfuse_observability import LangfuseObservability


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    if settings.chroma_mode == "server":
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))


@lru_cache
def get_embedding_service():
    settings = get_settings()
    return create_embedding_service(model_name=settings.embedding_model)


@lru_cache
def get_vector_store() -> VectorStoreBase:
    settings = get_settings()
    return ChromaVectorStore(
        chroma_client=get_chroma_client(),
        collection_name=settings.chroma_collection,
    )


@lru_cache
def get_chunker() -> TextChunker:
    settings = get_settings()
    return TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


@lru_cache
def get_llm_client() -> LLMClientBase:
    settings = get_settings()

    if settings.llm_backend == "anthropic":
        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.model_name,
            timeout=settings.llm_timeout,
        )

    raise ValueError(
        f"Unsupported LLM_BACKEND '{settings.llm_backend}'. "
        f"Supported values: 'anthropic'. "
        f"To add a new provider: create app/core/backends/<provider>.py, "
        f"implement LLMClientBase, and add an elif branch here."
    )


@lru_cache
def get_observability() -> ObservabilityBase:
    settings = get_settings()

    if not settings.eval_enabled:
        return NullObservability()

    return LangfuseObservability(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        anthropic_api_key=settings.anthropic_api_key,
        eval_model=settings.eval_model,
    )
