import argparse
import os
import random
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

TARGET = os.getenv("TARGET_COLUMN", "demand")
EXCLUDED_COLUMNS = {"timestamp", TARGET}

FEATURE_COLUMNS = [col for col in reference.columns if col not in EXCLUDED_COLUMNS]


def generate_data(mode):
    row = reference.sample(n=1).iloc[0]
    payload = {}

    # 1. Feature Generation Logic
    for column in FEATURE_COLUMNS:
        value = row[column]

        # Shift features ONLY in 'feature_drift' and 'all_drift' modes
        if mode in ["feature_drift", "all_drift"]:
            if pd.api.types.is_numeric_dtype(reference[column]):
                mean = reference[column].mean()
                std = reference[column].std()
                if pd.isna(std) or std == 0:
                    std = 1.0
                # Shift distribution by 5 standard deviations
                payload[column] = float(mean + 5 * std)
            else:
                payload[column] = value.item() if hasattr(value, "item") else value
        else:
            # Normal features directly from reference dataset
            payload[column] = value.item() if hasattr(value, "item") else value

    # 2. Attach metadata signal so label producer knows the mode
    payload["drift_mode"] = mode

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