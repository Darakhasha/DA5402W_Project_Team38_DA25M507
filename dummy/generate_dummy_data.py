import os
import numpy as np
import pandas as pd

np.random.seed(42)


def generate_datasets():
    n_training = 1000

    timestamps = pd.date_range(start="2026-01-01", periods=n_training, freq="h")

    pu_locations = np.random.randint(1, 264, size=n_training)
    avg_trip_distance = np.round(
        np.random.exponential(scale=3.0, size=n_training) + 0.5, 2
    )
    avg_fare_amount = np.round(
        avg_trip_distance * 3.5 + np.random.normal(2.5, 1.0, size=n_training), 2
    )

    hour = timestamps.hour
    day_of_week = timestamps.dayofweek
    is_weekend = (day_of_week >= 5).astype(int)

    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)

    demand = np.maximum(
        0,
        (
            20
            + 15 * ((hour >= 7) & (hour <= 9))
            + 25 * ((hour >= 17) & (hour <= 20))
            + 0.5 * avg_fare_amount
            + np.random.normal(0, 5, n_training)
        ),
    ).astype(int)

    training_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "PULocationID": pu_locations,
            "avg_trip_distance": avg_trip_distance,
            "avg_fare_amount": avg_fare_amount,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "demand": demand,
        }
    )

    # Save feature reference parquet for Pipelines 1, 2, 3, and 4
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    target_parquet_path = os.path.join(output_dir, "taxi_demand_features.parquet")
    training_data.to_parquet(target_parquet_path, index=False)
    print(f"[DATA] Successfully created reference parquet at {target_parquet_path}")

    # Save baseline prediction reference CSV for prediction drift calculation
    dummy_dir = "dummy"
    os.makedirs(dummy_dir, exist_ok=True)
    ref_preds_path = os.path.join(dummy_dir, "reference_predictions.csv")
    pd.DataFrame({"prediction": demand}).to_csv(ref_preds_path, index=False)
    print(f"[DATA] Successfully created prediction reference CSV at {ref_preds_path}")


if __name__ == "__main__":
    generate_datasets()