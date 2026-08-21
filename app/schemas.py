from __future__ import annotations

import datetime as dt
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    location_id: int = Field(
        ...,
        ge=1,
        description="Taxi pickup zone ID (PULocationID)",
        examples=[42],
    )

    timestamp: dt.datetime = Field(
        ...,
        description="Timestamp for which demand is predicted",
        examples=["2026-08-20T18:30:00"],
    )

    avg_trip_distance: Optional[float] = Field(
        2.5,
        ge=0.0,
        description="Historical or estimated average trip distance in miles",
        examples=[3.2],
    )

    avg_fare_amount: Optional[float] = Field(
        15.0,
        ge=0.0,
        description="Historical or estimated average fare amount in USD",
        examples=[18.5],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "location_id": 42,
                    "timestamp": "2026-08-20T18:30:00",
                    "avg_trip_distance": 3.2,
                    "avg_fare_amount": 18.5,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    location_id: int
    timestamp: dt.datetime
    predicted_demand: float
    model_name: str
    model_version: str
    horizon_minutes: int
    request_id: str
    features: dict[str, float | int | None]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    horizon_minutes: int
    feature_columns: list[str]
    champion_metrics: dict


class EngineeredFeatureRecord(BaseModel):
    location_id: int
    timestamp: dt.datetime
    avg_trip_distance: float
    avg_fare_amount: float
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    sin_hour: float
    cos_hour: float