from fastapi import FastAPI
import random

app = FastAPI(
    title="Taxi Demand Prediction API",
    description="Dummy API for Pipeline 4 development",
    version="1.0"
)


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
def predict():
    prediction = random.randint(100, 300)

    return {
        "predicted_demand": prediction
    }