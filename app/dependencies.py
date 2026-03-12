from functools import lru_cache

import chromadb

from app.settings import Settings
from app.core.chunker import TextChunker
from app.core.embeddings import create_embedding_service
from app.core.vector_store import DocsVectorStore
from app.core.llm_client import AnthropicClient


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))


@lru_cache
def get_embedding_service():
    settings = get_settings()
    return create_embedding_service(model_name=settings.embedding_model)


@lru_cache
def get_vector_store() -> DocsVectorStore:
    return DocsVectorStore(chroma_client=get_chroma_client())


@lru_cache
def get_chunker() -> TextChunker:
    settings = get_settings()
    return TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


@lru_cache
def get_llm_client() -> AnthropicClient:
    settings = get_settings()
    return AnthropicClient(api_key=settings.anthropic_api_key, model=settings.model_name)
