SAMPLE_DOC = {
    "source_id": "guide-01",
    "title": "Getting Started",
    "content": (
        "This guide explains how to get started with the API. "
        "First, index your documentation using the POST /index endpoint. "
        "Then query it with POST /ask. "
        "You can manage sources with GET /sources and DELETE /sources/{id}."
    ),
}


def test_list_sources_empty(client):
    response = client.get("/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert data["total"] == 0


def test_list_sources_after_index(client):
    client.post("/index", json=SAMPLE_DOC)
    response = client.get("/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    source = data["sources"][0]
    assert source["source_id"] == "guide-01"
    assert source["title"] == "Getting Started"
    assert source["chunks_count"] >= 1


def test_delete_source(client):
    client.post("/index", json=SAMPLE_DOC)
    response = client.delete("/sources/guide-01")
    assert response.status_code == 200
    assert response.json()["deleted_chunks"] >= 1

    # Confirm it's gone
    response = client.get("/sources")
    assert response.json()["total"] == 0


def test_delete_nonexistent_source_returns_404(client):
    response = client.delete("/sources/does-not-exist")
    assert response.status_code == 404
