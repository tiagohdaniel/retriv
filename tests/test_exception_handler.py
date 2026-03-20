"""Test that unhandled errors return a generic 500 without exposing internals."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client_with_bomb():
    """Register a route that raises an unhandled Exception."""

    @app.get("/_test_bomb")
    def _bomb():
        raise RuntimeError("internal secret data / stack trace")

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Remove the temporary route
    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/_test_bomb"]


def test_unhandled_exception_returns_500_generic(client_with_bomb):
    response = client_with_bomb.get("/_test_bomb")
    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal server error"}
    assert "secret" not in response.text
    assert "Traceback" not in response.text
