"""
Tests for Darshita's deployment pipeline.

Run with: pytest tests/ -v
Requires models/model.joblib to exist (run `python train_model.py` first) —
the fixture below trains it automatically if missing so CI can run cold.
"""
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def ensure_model_trained():
    if not os.path.exists("models/model.joblib"):
        subprocess.run([sys.executable, "train_model.py"], check=True)
    yield


@pytest.fixture(scope="session")
def client():
    from app.main import app

    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    resp = client.get("/model/info")
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
    assert body["location_id"] == 42
    assert body["predicted_demand"] >= 0
    assert "request_id" in body
    assert "model_name" in body


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
    assert resp.json()["predicted_demand"] >= 0


def test_predict_invalid_location_id(client):
    resp = client.post(
        "/predict",
        json={"location_id": -1, "timestamp": "2026-06-22T18:30:00"},
    )
    assert resp.status_code == 422  # pydantic validation error (ge=0)


def test_predictions_are_logged(client, tmp_path=None):
    client.post(
        "/predict",
        json={"location_id": 5, "timestamp": "2026-06-22T18:30:00"},
    )
    assert os.path.exists("logs/predictions.log")
    with open("logs/predictions.log") as f:
        lines = f.readlines()
    assert len(lines) > 0
    import json as _json

    last = _json.loads(lines[-1])
    assert "predicted_demand" in last
    assert "request_id" in last


def test_swagger_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
