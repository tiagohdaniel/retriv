SAMPLE_DOC = {
    "source_id": "python-docs",
    "title": "Python asyncio",
    "content": (
        "asyncio is a library to write concurrent code using async/await syntax. "
        "It provides an event loop that runs coroutines. "
        "You can use asyncio.gather to run multiple coroutines concurrently. "
        "The event loop schedules coroutines and handles I/O operations efficiently."
    ),
}


def test_ask_with_indexed_content(client):
    client.post("/index", json=SAMPLE_DOC)
    # max_distance=2.0 disables the threshold — needed because HashEmbedding
    # (used in tests) produces random vectors unrelated to text meaning.
    response = client.post("/ask", json={"question": "What is asyncio?", "top_k": 2, "max_distance": 2.0})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Mocked answer from LLM."
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) >= 1


def test_ask_without_indexed_content_returns_fallback(client):
    """No documents indexed → LLM is NOT called, fallback answer returned."""
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 200
    data = response.json()
    assert "No relevant documentation found" in data["answer"]
    assert data["sources"] == []


def test_ask_with_source_id_filter(client):
    client.post("/index", json=SAMPLE_DOC)
    response = client.post("/ask", json={
        "question": "What is asyncio?",
        "source_ids": ["python-docs"],
        "max_distance": 2.0,
    })
    assert response.status_code == 200


def test_ask_requires_question(client):
    response = client.post("/ask", json={"top_k": 3})
    assert response.status_code == 422
