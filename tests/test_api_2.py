import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.model_loader import LoadedModel

# Prevent Kafka connections from hanging test runs
os.environ["TESTING"] = "true"


@pytest.fixture(scope="session", autouse=True)
def mock_mlflow_model():
    """Mock the model loader for unit testing the API endpoints."""
    class DummyModel:
        def predict(self, df):
            return [42.0] * len(df)

    dummy_loaded_model = LoadedModel(
        model=DummyModel(),
        model_name="TaxiDemandModel",
        model_version="v1",
        model_uri="models:/mock"
    )
    
    with patch("app.model_loader.get_model", return_value=dummy_loaded_model):
        yield


@pytest.fixture(scope="session")
def client():
    from app.main import app
    return TestClient(app)


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data or "status" in data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ["healthy", "ok"]


def test_predict(client):
    payload = {
        "location_id": 1,
        "timestamp": "2026-08-21T18:30:00"
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    pred_val = data.get("prediction", data.get("predicted_demand"))

    assert pred_val is not None
    assert pred_val >= 0