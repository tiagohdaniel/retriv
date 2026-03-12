"""
Test fixtures: replaces production dependencies with isolated test doubles.

- EphemeralClient + UUID collection name: isolated per test
  (EphemeralClient shares process memory, so collection names must be unique)
- HashEmbedding: no model download, deterministic vectors
- AsyncMock LLM: no API key, predictable output
"""
import uuid
import pytest
import chromadb
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import (
    get_embedding_service,
    get_vector_store,
    get_chunker,
    get_llm_client,
)
from app.core.embeddings import HashEmbedding
from app.core.vector_store import DocsVectorStore
from app.core.chunker import TextChunker


def _mock_llm():
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value={
        "answer": "Mocked answer from LLM.",
        "tokens_used": 42,
        "model": "mock-model",
    })
    return mock


@pytest.fixture
def client():
    collection_name = f"test_{uuid.uuid4().hex}"
    ephemeral = chromadb.EphemeralClient()
    embedding = HashEmbedding()
    vector_store = DocsVectorStore(chroma_client=ephemeral, collection_name=collection_name)
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    llm = _mock_llm()

    app.dependency_overrides = {
        get_embedding_service: lambda: embedding,
        get_vector_store: lambda: vector_store,
        get_chunker: lambda: chunker,
        get_llm_client: lambda: llm,
    }

    with TestClient(app) as c:
        yield c

    app.dependency_overrides = {}
