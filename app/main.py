from fastapi import FastAPI
import random
from prometheus_fastapi_instrumentator import Instrumentator
from app.kafka_producer import publish_prediction_request

app = FastAPI(
    title="Taxi Demand Prediction API",
    description="Dummy API for Pipeline 4 development",
    version="1.0"
)

Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    return {
        "message": "Taxi Demand API is Running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(data: dict):
    prediction = random.randint(100, 300)
   # publish_prediction_request(data.model_dump())

    publish_prediction_request(data)

    return {
        "predicted_demand": prediction
    }