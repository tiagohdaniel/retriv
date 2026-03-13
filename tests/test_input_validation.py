"""Tests for input size and format validation on IndexRequest and AskRequest."""


def test_index_content_too_large_returns_422(client):
    response = client.post("/index", json={
        "source_id": "test",
        "title": "Test",
        "content": "x" * 500_001,
    })
    assert response.status_code == 422


def test_index_content_minimum_enforced(client):
    response = client.post("/index", json={
        "source_id": "test",
        "title": "Test",
        "content": "short",       # below min_length=10
    })
    assert response.status_code == 422


def test_index_source_id_too_long_returns_422(client):
    response = client.post("/index", json={
        "source_id": "x" * 201,   # above max_length=200
        "title": "Test",
        "content": "valid content here for indexing",
    })
    assert response.status_code == 422


def test_ask_question_too_long_returns_422(client):
    response = client.post("/ask", json={"question": "x" * 1_001})
    assert response.status_code == 422


def test_ask_question_minimum_enforced(client):
    response = client.post("/ask", json={"question": "ok"})   # below min_length=5
    assert response.status_code == 422
