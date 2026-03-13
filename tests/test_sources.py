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


def _make_doc(n: int) -> dict:
    return {
        "source_id": f"doc-{n:02d}",
        "title": f"Document {n}",
        "content": f"This is the full content of document number {n}. " * 5,
    }


def test_list_sources_empty(client):
    response = client.get("/sources")
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert data["total"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 50


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


def test_pagination_page_and_page_size_in_response(client):
    client.post("/index", json=SAMPLE_DOC)
    response = client.get("/sources?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 10


def test_pagination_second_page_is_empty_when_only_one_source(client):
    client.post("/index", json=SAMPLE_DOC)
    response = client.get("/sources?page=2&page_size=1")
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert data["total"] == 1


def test_pagination_splits_sources_across_pages(client):
    for i in range(1, 4):
        client.post("/index", json=_make_doc(i))

    page1 = client.get("/sources?page=1&page_size=2").json()
    page2 = client.get("/sources?page=2&page_size=2").json()

    assert len(page1["sources"]) == 2
    assert len(page2["sources"]) == 1
    assert page1["total"] == 3
    assert page2["total"] == 3

    ids_page1 = {s["source_id"] for s in page1["sources"]}
    ids_page2 = {s["source_id"] for s in page2["sources"]}
    assert ids_page1.isdisjoint(ids_page2)


def test_pagination_invalid_page_size_returns_422(client):
    response = client.get("/sources?page_size=0")
    assert response.status_code == 422

    response = client.get("/sources?page_size=201")
    assert response.status_code == 422
