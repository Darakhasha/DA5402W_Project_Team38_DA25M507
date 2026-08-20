from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Prediction Request Schema
# Receives engineered features from Pipeline 2
# ---------------------------------------------------------------------
class PredictRequest(BaseModel):
    # Metadata
    location_id: int = Field(
        ...,
        ge=0,
        description="Taxi pickup zone / location ID",
        examples=[42],
    )

    timestamp: dt.datetime = Field(
        ...,
        description="Timestamp for which demand is predicted",
        examples=["2026-08-20T18:30:00"],
    )

    # Engineered features from Spark Pipeline
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day",
        examples=[18],
    )

    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week (Monday=0, Sunday=6)",
        examples=[3],
    )

    temperature: Optional[float] = Field(
        None,
        description="Temperature in Celsius",
        examples=[28.5],
    )

    rain: Optional[float] = Field(
        None,
        ge=0,
        description="Rainfall / precipitation value",
        examples=[2.4],
    )

    traffic_index: Optional[float] = Field(
        None,
        ge=0,
        description="Traffic congestion index",
        examples=[65.0],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "location_id": 42,
                    "timestamp": "2026-08-20T18:30:00",
                    "hour": 18,
                    "day_of_week": 3,
                    "temperature": 28.5,
                    "rain": 2.4,
                    "traffic_index": 65.0,
                }
            ]
        }
    }


# ---------------------------------------------------------------------
# Prediction Response Schema
# Returned by FastAPI after model inference
# ---------------------------------------------------------------------
class PredictResponse(BaseModel):
    location_id: int
    timestamp: dt.datetime

    predicted_demand: float

    model_name: str
    model_version: str
    horizon_minutes: int

    request_id: str

    # Features used for prediction (useful for monitoring & debugging)
    features: dict[str, float | int | None]


# ---------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------
# Model Information
# ---------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    horizon_minutes: int
    feature_columns: list[str]
    champion_metrics: dict


# ---------------------------------------------------------------------
# Optional Schema for Spark Pipeline Output
# Can be used internally between Pipeline 2 and Pipeline 3
# ---------------------------------------------------------------------
class EngineeredFeatureRecord(BaseModel):
    location_id: int
    timestamp: dt.datetime
    hour: int
    day_of_week: int
    temperature: Optional[float]
    rain: Optional[float]
    traffic_index: Optional[float]