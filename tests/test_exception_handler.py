"""Testa que erros não tratados retornam 500 genérico sem expor internals."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client_with_bomb():
    """Registra uma rota que lança uma Exception não tratada."""

    @app.get("/_test_bomb")
    def _bomb():
        raise RuntimeError("internal secret data / stack trace")

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Remove a rota temporária
    app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/_test_bomb"]


def test_unhandled_exception_returns_500_generic(client_with_bomb):
    response = client_with_bomb.get("/_test_bomb")
    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal server error"}
    assert "secret" not in response.text
    assert "Traceback" not in response.text
