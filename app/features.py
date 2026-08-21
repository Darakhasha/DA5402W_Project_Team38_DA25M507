from __future__ import annotations

import datetime as dt
from typing import Optional
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "PULocationID",
    "avg_trip_distance",
    "avg_fare_amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "sin_hour",
    "cos_hour",
]


def build_feature_row(
    location_id: int,
    timestamp: dt.datetime,
    avg_trip_distance: Optional[float] = None,
    avg_fare_amount: Optional[float] = None,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame synchronized with training features."""
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()  # Monday=0, Sunday=6
    is_weekend = int(day_of_week >= 5)

    sin_hour = np.sin(2 * np.pi * hour_of_day / 24.0)
    cos_hour = np.cos(2 * np.pi * hour_of_day / 24.0)

    row = {
        "PULocationID": int(location_id),
        "avg_trip_distance": float(
            avg_trip_distance if avg_trip_distance is not None else 2.5
        ),
        "avg_fare_amount": float(
            avg_fare_amount if avg_fare_amount is not None else 15.0
        ),
        "hour_of_day": int(hour_of_day),
        "day_of_week": int(day_of_week),
        "is_weekend": int(is_weekend),
        "sin_hour": float(sin_hour),
        "cos_hour": float(cos_hour),
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)