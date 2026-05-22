from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint_returns_running_status() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "AI Body Engine"
    assert response.json()["status"] == "running"
    assert response.json()["version"] == "0.1.0"


def test_health_endpoint_returns_phase_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "ai-body-engine"
    assert response.json()["phase"] == "phase-1-skeleton"
