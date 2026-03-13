"""Tests for the Prometheus /metrics endpoint."""


def test_metrics_endpoint_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_content_type_is_prometheus(client):
    response = client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


def test_metrics_contains_http_metrics(client):
    # trigger a request so counters are non-zero
    client.get("/health")
    response = client.get("/metrics")
    assert "http_requests_total" in response.text


def test_metrics_no_auth_required(client):
    """Prometheus scraper must reach /metrics without an API key."""
    response = client.get("/metrics")
    assert response.status_code != 401


def test_metrics_contains_custom_metrics_after_index(client):
    client.post("/index", json={
        "source_id": "metrics-test",
        "title": "Metrics Test",
        "content": "content used to verify metrics are emitted after indexing documents",
    })
    response = client.get("/metrics")
    assert "retriv_documents_indexed_total" in response.text
    assert "retriv_chunks_indexed_total" in response.text
