from fastapi import FastAPI
import random
from prometheus_fastapi_instrumentator import Instrumentator
from app.kafka_producer import (
    publish_inference,
    publish_feedback,
)
import uuid
from datetime import datetime, timezone

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
    request_id = str(uuid.uuid4())

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    publish_inference(
        request_id=request_id,
        timestamp=timestamp,
        features=data,
        prediction=prediction,
    )

    return {
        "request_id": request_id,
        "prediction": prediction,
    }

# @app.post("/feedback")
# def feedback(
#     request_id: str,
#     actual: float,
# ):

#     publish_feedback(
#         request_id=request_id,
#         actual=actual,
#     )

#     return {
#         "status": "feedback recorded",
#         "request_id": request_id,
#         "actual": actual,
#     }