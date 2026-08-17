from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Taxi Demand API is Running"


def test_predict():
    response = client.post("/predict")

    assert response.status_code == 200

    data = response.json()

    assert "predicted_demand" in data

    assert 100 <= data["predicted_demand"] <= 300