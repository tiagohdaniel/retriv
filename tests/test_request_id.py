"""Tests for the RequestLoggingMiddleware and X-Request-ID propagation."""


def test_response_includes_request_id(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers


def test_client_request_id_is_echoed(client):
    """When the client sends X-Request-ID, the same value is returned."""
    custom_id = "test-correlation-id-123"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers["x-request-id"] == custom_id


def test_server_generates_request_id_when_absent(client):
    """When no X-Request-ID header is sent, the server generates a UUID."""
    response = client.get("/health")
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 36  # UUID4 format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    assert request_id.count("-") == 4
