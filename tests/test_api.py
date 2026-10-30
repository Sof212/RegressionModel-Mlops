
from fastapi.testclient import TestClient

from src.app.app import app

client = TestClient(app)


def test_root_endpoint():
    """Vérifie que l'API est opérationnelle."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_validation():
    """Vérifie la validation des données envoyées à l'API."""
    response = client.post("/predict", json={})

    assert response.status_code == 422
