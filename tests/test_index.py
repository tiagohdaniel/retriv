SAMPLE_DOC = {
    "source_id": "fastapi-docs",
    "title": "FastAPI Introduction",
    "content": (
        "FastAPI is a modern, fast web framework for building APIs with Python. "
        "It is based on standard Python type hints and provides automatic OpenAPI docs. "
        "FastAPI supports async/await natively and has excellent performance. "
        "It uses Pydantic for data validation and serialization."
    ),
}


def test_index_returns_chunk_count(client):
    response = client.post("/index", json=SAMPLE_DOC)
    assert response.status_code == 200
    data = response.json()
    assert data["source_id"] == "fastapi-docs"
    assert data["chunks_indexed"] >= 1


def test_index_is_idempotent(client):
    """Re-indexing the same source_id replaces existing chunks."""
    client.post("/index", json=SAMPLE_DOC)
    response = client.post("/index", json=SAMPLE_DOC)
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] >= 1


def test_index_requires_source_id(client):
    response = client.post("/index", json={"title": "X", "content": "Y"})
    assert response.status_code == 422


def test_index_requires_content(client):
    response = client.post("/index", json={"source_id": "x", "title": "X"})
    assert response.status_code == 422
