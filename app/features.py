"""
Feature engineering shared by training (train_model.py) and serving (app/main.py).

In the real team pipeline this module would read the same feature schema that
Zeba's Kafka/Spark ingestion + Airflow DAG produce and version with DVC. Here it
is reimplemented locally so Darshita's deployment pipeline is fully self-contained
and testable without the other three pipelines running.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib

    _INDIA_HOLIDAYS = holidays_lib.India()
except Exception:  # pragma: no cover - holidays lib is optional at runtime
    _INDIA_HOLIDAYS = {}

FEATURE_COLUMNS = [
    "location_id",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_rush_hour",
    "temperature_c",
    "precipitation_mm",
    "rolling_avg_1h",
    "rolling_avg_24h",
]


def _is_rush_hour(hour: int) -> int:
    return int((7 <= hour <= 10) or (17 <= hour <= 20))


def build_feature_row(
    location_id: int,
    timestamp: dt.datetime,
    is_holiday: Optional[bool] = None,
    temperature_c: Optional[float] = None,
    precipitation_mm: Optional[float] = None,
    rolling_avg_1h: Optional[float] = None,
    rolling_avg_24h: Optional[float] = None,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for the given (location, timestamp).

    Mirrors the "Time Features" + weather join described in the project brief:
    hour of day, day of week, month, weekend flag, holiday flag, rush-hour flag,
    plus rolling demand windows and a weather join.
    """
    if is_holiday is None:
        is_holiday = timestamp.date() in _INDIA_HOLIDAYS

    if temperature_c is None:
        # Simple seasonal default so the endpoint works with no weather feed attached.
        temperature_c = 25 + 8 * np.sin(2 * np.pi * (timestamp.timetuple().tm_yday / 365))

    if precipitation_mm is None:
        precipitation_mm = 0.0

    # In production these two rolling features come from Zeba's Spark Streaming
    # aggregation window (avg rides in this zone over last 1h / 24h). Here we fall
    # back to a location-conditioned heuristic seed so the model has something
    # sensible to consume when called standalone.
    if rolling_avg_1h is None:
        rolling_avg_1h = 20 + (location_id % 50)
    if rolling_avg_24h is None:
        rolling_avg_24h = 20 + (location_id % 50)

    row = {
        "location_id": location_id,
        "hour": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "month": timestamp.month,
        "is_weekend": int(timestamp.weekday() >= 5),
        "is_holiday": int(bool(is_holiday)),
        "is_rush_hour": _is_rush_hour(timestamp.hour),
        "temperature_c": float(temperature_c),
        "precipitation_mm": float(precipitation_mm),
        "rolling_avg_1h": float(rolling_avg_1h),
        "rolling_avg_24h": float(rolling_avg_24h),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
