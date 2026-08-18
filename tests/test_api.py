from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Taxi Demand API is Running"

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    payload = {
        "pickup_location_id": 1,
        "passenger_count": 2
    }
    response = client.post("/predict",json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert 100 <= data["prediction"] <= 300