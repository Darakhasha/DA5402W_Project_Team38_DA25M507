"""
Pipeline 3 - Deployment API

Responsibilities:
    1. Load champion model from MLflow
    2. FastAPI
    3. POST /predict
    4. Swagger documentation
    5. Logging
    6. Publish inference event to Pipeline 4 Kafka
    7. Prometheus API metrics
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import timezone

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from app.features import build_feature_row
from app.kafka_producer import publish_inference
from app.logging_config import app_logger, log_prediction
from app.model_loader import (
    ModelNotLoadedError,
    get_model,
    reload_model,
)
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


# =========================================================
# MODEL LOADING AT APPLICATION STARTUP
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:

        loaded = get_model()

        app_logger.info(
            "Champion model loaded successfully: "
            "%s version=%s",
            loaded.model_name,
            loaded.model_version,
        )

    except ModelNotLoadedError as exc:

        app_logger.error(
            "Champion model could not be loaded: %s",
            exc,
        )

    yield


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Taxi Demand Forecasting API",
    description=(
        "Pipeline 3 Deployment API. "
        "Loads the champion model from MLflow and "
        "serves taxi-demand predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# PROMETHEUS
# =========================================================

Instrumentator().instrument(app).expose(app)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Taxi Demand Prediction API is running",
        "pipeline": "Pipeline 3 - Deployment",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
)
def health():

    try:

        loaded = get_model()

        return HealthResponse(
            status="healthy",
            model_loaded=True,
        )

    except ModelNotLoadedError:

        return HealthResponse(
            status="degraded",
            model_loaded=False,
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["operations"],
)
def model_info():

    try:

        loaded = get_model()

    except ModelNotLoadedError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return ModelInfoResponse(
        model_name=loaded.model_name,
        model_version=loaded.model_version,
        horizon_minutes=30,
        feature_columns=[],
        champion_metrics={},
    )


# =========================================================
# MODEL RELOAD
# =========================================================

@app.post(
    "/model/reload",
    tags=["operations"],
)
def model_reload():

    """
    Reload the model currently marked as @champion in MLflow.

    Pipeline 2 can promote a new model version and then
    Pipeline 3 can call this endpoint to load it.
    """

    try:

        loaded = reload_model()

        app_logger.info(
            "Champion model reloaded: "
            "%s version=%s",
            loaded.model_name,
            loaded.model_version,
        )

        return {
            "status": "reloaded",
            "model_name": loaded.model_name,
            "model_version": loaded.model_version,
        }

    except ModelNotLoadedError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
)
def predict(payload: PredictRequest):

    # -----------------------------------------------------
    # STEP 1: REQUEST ID
    # -----------------------------------------------------

    request_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    app_logger.info(
        "[%s] Prediction request received",
        request_id,
    )


    # -----------------------------------------------------
    # STEP 2: GET CHAMPION MODEL
    # -----------------------------------------------------

    try:

        loaded = get_model()

    except ModelNotLoadedError as exc:

        app_logger.error(
            "[%s] Champion model unavailable: %s",
            request_id,
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail="Champion model is unavailable",
        ) from exc


    # -----------------------------------------------------
    # STEP 3: BUILD INFERENCE FEATURES
    # -----------------------------------------------------

    try:

        features = build_feature_row(
            location_id=payload.location_id,
            timestamp=payload.timestamp,
            is_holiday=payload.is_holiday,
            temperature_c=payload.temperature_c,
            precipitation_mm=payload.precipitation_mm,
        )

    except Exception as exc:

        app_logger.exception(
            "[%s] Feature preparation failed",
            request_id,
        )

        raise HTTPException(
            status_code=400,
            detail=f"Feature preparation error: {exc}",
        ) from exc


    # -----------------------------------------------------
    # STEP 4: MODEL PREDICTION
    # -----------------------------------------------------

    try:

        raw_prediction = loaded.model.predict(features)[0]

        predicted_demand = max(
            0.0,
            round(float(raw_prediction), 2),
        )

    except Exception as exc:

        app_logger.exception(
            "[%s] Model inference failed",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Model inference error: {exc}",
        ) from exc


    # -----------------------------------------------------
    # STEP 5: CALCULATE LATENCY
    # -----------------------------------------------------

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )


    # -----------------------------------------------------
    # STEP 6: LOG PREDICTION
    # -----------------------------------------------------

    timestamp = payload.timestamp

    if timestamp.tzinfo is None:

        timestamp = timestamp.replace(
            tzinfo=timezone.utc
        )


    prediction_record = {

        "event_type": "prediction",

        "request_id": request_id,

        "location_id": payload.location_id,

        "timestamp": timestamp.isoformat(),

        "predicted_demand": predicted_demand,

        "model_name": loaded.model_name,

        "model_version": loaded.model_version,

        "latency_ms": latency_ms,

    }

    log_prediction(prediction_record)


    app_logger.info(
        "[%s] Prediction completed: "
        "location=%s prediction=%.2f latency=%.2fms",
        request_id,
        payload.location_id,
        predicted_demand,
        latency_ms,
    )


    # -----------------------------------------------------
    # STEP 7: SEND INFERENCE EVENT TO PIPELINE 4
    # -----------------------------------------------------

    try:

        publish_inference(

            request_id=request_id,

            timestamp=timestamp.isoformat(),

            features=payload.model_dump(
                mode="json"
            ),

            prediction=predicted_demand,

        )

    except Exception as exc:

        # IMPORTANT:
        # Do not make the prediction itself fail just because
        # monitoring/Kafka is temporarily unavailable.

        app_logger.exception(
            "[%s] Kafka publishing failed: %s",
            request_id,
            exc,
        )


    # -----------------------------------------------------
    # STEP 8: RESPONSE
    # -----------------------------------------------------

    return PredictResponse(

        location_id=payload.location_id,

        timestamp=payload.timestamp,

        predicted_demand=predicted_demand,

        model_name=loaded.model_name,

        model_version=loaded.model_version,

        horizon_minutes=30,

        request_id=request_id,

    )