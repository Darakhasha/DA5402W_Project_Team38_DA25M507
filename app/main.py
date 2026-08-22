from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import timezone

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from app.features import FEATURE_COLUMNS, build_feature_row
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # try:
    #     loaded = get_model()
    #     if loaded is not None:
    #         app_logger.info(
    #             "Champion model loaded successfully: %s version=%s",
    #             loaded.model_name,
    #             loaded.model_version,
    #         )
    #     else:
    #         app_logger.warning(
    #             "Champion model could not be loaded at startup. API is running in degraded state."
    #         )
    # except ModelNotLoadedError as exc:
    #     app_logger.error(
    #         "Champion model could not be loaded at startup: %s",
    #         exc,
    #     )
    app_logger.info("Starting FastAPI application...")
    yield


app = FastAPI(
    title="Taxi Demand Forecasting API",
    description=(
        "Pipeline 3 Deployment API. "
        "Loads the champion model from MLflow and serves taxi-demand predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def home():
    return {
        "message": "Taxi Demand Prediction API is running",
        "pipeline": "Pipeline 3 - Deployment",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
)
def health():
    try:
        get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
        )
    except ModelNotLoadedError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
        )


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
        horizon_minutes=60,
        feature_columns=FEATURE_COLUMNS,
        champion_metrics={},
    )


@app.post(
    "/model/reload",
    tags=["operations"],
)
def model_reload():
    try:
        loaded = reload_model()
        app_logger.info(
            "Champion model reloaded: %s version=%s",
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


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
)
def predict(payload: PredictRequest):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    app_logger.info("[%s] Prediction request received", request_id)

    try:
        loaded = get_model()
    except ModelNotLoadedError as exc:
        app_logger.error("[%s] Champion model unavailable: %s", request_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Champion model is unavailable",
        ) from exc

    try:
        features_df = build_feature_row(
            location_id=payload.location_id,
            timestamp=payload.timestamp,
            avg_trip_distance=payload.avg_trip_distance,
            avg_fare_amount=payload.avg_fare_amount,
        )
    except Exception as exc:
        app_logger.exception("[%s] Feature preparation failed", request_id)
        raise HTTPException(
            status_code=400,
            detail=f"Feature preparation error: {exc}",
        ) from exc

    try:
        raw_prediction = loaded.model.predict(features_df)[0]
        predicted_demand = max(0.0, round(float(raw_prediction), 2))
    except Exception as exc:
        app_logger.exception("[%s] Model inference failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Model inference error: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    timestamp = (
        payload.timestamp
        if payload.timestamp.tzinfo
        else payload.timestamp.replace(tzinfo=timezone.utc)
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
        "[%s] Prediction completed: location=%s prediction=%.2f latency=%.2fms",
        request_id,
        payload.location_id,
        predicted_demand,
        latency_ms,
    )

    try:
        publish_inference(
            request_id=request_id,
            timestamp=timestamp.isoformat(),
            features=features_df.to_dict(orient="records")[0],
            prediction=predicted_demand,
        )
    except Exception as exc:
        app_logger.exception("[%s] Kafka publishing failed: %s", request_id, exc)

    return PredictResponse(
        location_id=payload.location_id,
        timestamp=payload.timestamp,
        predicted_demand=predicted_demand,
        model_name=loaded.model_name,
        model_version=loaded.model_version,
        horizon_minutes=60,
        request_id=request_id,
        features=features_df.to_dict(orient="records")[0],
    )