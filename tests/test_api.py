import os
from fastapi.testclient import TestClient

# Prevent Kafka connections from hanging test runs
os.environ["TESTING"] = "true"

from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data or "status" in data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ["healthy", "ok"]


def test_predict():
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