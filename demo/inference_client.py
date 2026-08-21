import argparse
import datetime as dt
import os
import time
import pandas as pd
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000/predict")
parser.add_argument("--iterations", type=int, default=10)
parser.add_argument(
    "--mode",
    choices=["normal", "feature_drift", "performance_drift", "all_drift"],
    default="normal",
    help="Select the test case to simulate.",
)
parser.add_argument("--reference", default="data/processed/taxi_demand_features.parquet")

args = parser.parse_args()

reference = pd.read_parquet(args.reference)

def generate_data(mode):
    row = reference.sample(n=1).iloc[0]
    
    # Extract Location ID (handles both PULocationID and location_id)
    loc_id = int(row["PULocationID"]) if "PULocationID" in row else int(row.get("location_id", 1))
    
    # Base feature values
    avg_dist = float(row.get("avg_trip_distance", 2.5))
    avg_fare = float(row.get("avg_fare_amount", 15.0))

    # Apply synthetic drift if requested
    if mode in ["feature_drift", "all_drift"]:
        avg_dist = float(avg_dist + 5 * (reference["avg_trip_distance"].std() or 1.0))
        avg_fare = float(avg_fare + 5 * (reference["avg_fare_amount"].std() or 5.0))

    # Construct payload matching app.schemas.PredictRequest
    payload = {
        "location_id": max(1, loc_id),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "avg_trip_distance": max(0.1, avg_dist),
        "avg_fare_amount": max(1.0, avg_fare),
    }

    return payload

print(f"Starting Inference Client in [{args.mode.upper()}] mode...\n")

for i in range(args.iterations):
    payload = generate_data(args.mode)

    response = requests.post(args.url, json=payload, timeout=10)
    response.raise_for_status()

    result = response.json()

    print(f"Iteration {i + 1}/{args.iterations}")
    print(f"Mode     : {args.mode}")
    print(f"Input    : {payload}")
    print(f"Response : {result}\n")

    time.sleep(1)