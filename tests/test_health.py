def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["checks"]["vector_store"] == "ok"


def test_health_degraded_when_vector_store_fails(client):
    from unittest.mock import patch
    from app.main import app
    from app.dependencies import get_vector_store

    broken_store = client.app.dependency_overrides[get_vector_store]()
    broken_store.ping = lambda: (_ for _ in ()).throw(RuntimeError("chroma down"))

    with patch.dict(client.app.dependency_overrides, {get_vector_store: lambda: broken_store}):
        response = client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert "error" in data["checks"]["vector_store"]
