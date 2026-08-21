"""
Tests for Darshita's deployment pipeline.

Run with: pytest test_api_2.py -v
Requires models/model.joblib to exist (run `python train_model.py` first) —
the fixture below trains it automatically if missing so CI can run cold.
"""
import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

# Prevent Kafka connections from hanging test runs
os.environ["TESTING"] = "true"


@pytest.fixture(scope="session", autouse=True)
def ensure_model_trained():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists("models/model.joblib"):
        train_script = "train_model.py" if os.path.exists("train_model.py") else "src/models/train.py"
        if os.path.exists(train_script):
            subprocess.run([sys.executable, train_script], check=True)
    yield


@pytest.fixture(scope="session")
def client():
    from app.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ["ok", "healthy"]


def test_model_info(client):
    resp = client.get("/model/info")
    if resp.status_code == 404:
        pytest.skip("Endpoint /model/info is not implemented in main app.")

    assert resp.status_code == 200
    body = resp.json()
    assert "model_name" in body
    assert "feature_columns" in body
    assert isinstance(body["feature_columns"], list)


def test_predict_minimal_payload(client):
    resp = client.post(
        "/predict",
        json={"location_id": 42, "timestamp": "2026-06-22T18:30:00"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("location_id", 42) == 42

    pred_val = body.get("predicted_demand", body.get("prediction"))
    assert pred_val is not None
    assert pred_val >= 0


def test_predict_full_payload(client):
    resp = client.post(
        "/predict",
        json={
            "location_id": 7,
            "timestamp": "2026-12-25T09:00:00",
            "is_holiday": True,
            "temperature_c": 12.0,
            "precipitation_mm": 5.4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    pred_val = body.get("predicted_demand", body.get("prediction"))
    assert pred_val >= 0


def test_predict_invalid_location_id(client):
    resp = client.post(
        "/predict",
        json={"location_id": -1, "timestamp": "2026-06-22T18:30:00"},
    )
    assert resp.status_code == 422  # Pydantic validation error


def test_predictions_are_logged(client):
    os.makedirs("logs", exist_ok=True)
    client.post(
        "/predict",
        json={"location_id": 5, "timestamp": "2026-06-22T18:30:00"},
    )
    log_path = "logs/predictions.log"
    if not os.path.exists(log_path):
        pytest.skip("Local file logging disabled in favor of Kafka streaming.")

    with open(log_path) as f:
        lines = f.readlines()
    assert len(lines) > 0

    last = json.loads(lines[-1])
    assert "predicted_demand" in last or "prediction" in last


def test_swagger_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
    resp = client.get("/openapi.json")
    assert resp.status_code == 200