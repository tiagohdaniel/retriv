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


def test_index_returns_job_id(client):
    response = client.post("/index", json=SAMPLE_DOC)
    assert response.status_code == 202
    data = response.json()
    assert data["source_id"] == "fastapi-docs"
    assert "job_id" in data
    assert data["status"] == "processing"


def test_index_job_completes(client):
    """Background task runs synchronously in TestClient — job is done after POST."""
    response = client.post("/index", json=SAMPLE_DOC)
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status = client.get(f"/index/status/{job_id}")
    assert status.status_code == 200
    data = status.json()
    assert data["status"] == "done"
    assert data["chunks_indexed"] >= 1


def test_index_is_idempotent(client):
    """Re-indexing the same source_id replaces existing chunks."""
    client.post("/index", json=SAMPLE_DOC)
    response = client.post("/index", json=SAMPLE_DOC)
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status = client.get(f"/index/status/{job_id}")
    assert status.json()["chunks_indexed"] >= 1


def test_index_requires_source_id(client):
    response = client.post("/index", json={"title": "X", "content": "Y"})
    assert response.status_code == 422


def test_index_requires_content(client):
    response = client.post("/index", json={"source_id": "x", "title": "X"})
    assert response.status_code == 422
