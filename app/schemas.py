from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    location_id: int = Field(..., ge=0, description="City zone / location ID", examples=[42])
    timestamp: dt.datetime = Field(
        ..., description="ISO-8601 timestamp to predict demand for", examples=["2026-06-22T18:30:00"]
    )
    is_holiday: Optional[bool] = Field(
        None, description="Override holiday flag; auto-detected (India calendar) if omitted"
    )
    temperature_c: Optional[float] = Field(
        None, description="Current temperature in Celsius; seasonal default used if omitted"
    )
    precipitation_mm: Optional[float] = Field(
        None, description="Precipitation in mm; defaults to 0 if omitted"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"location_id": 42, "timestamp": "2026-06-22T18:30:00"},
                {
                    "location_id": 7,
                    "timestamp": "2026-12-25T09:00:00",
                    "is_holiday": True,
                    "temperature_c": 12.0,
                    "precipitation_mm": 5.4,
                },
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    horizon_minutes: int
    feature_columns: list[str]
    champion_metrics: dict
