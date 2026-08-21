import csv
import json
import math
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
from kafka import KafkaConsumer

sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from scripts.s3_utils import upload_file
except ImportError:
    upload_file = None

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "taxi-events")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "taxi-monitoring")

OUTPUT_FILE = os.getenv("INFERENCE_OUTPUT_FILE", "monitoring/data/inference_data.csv")
REFERENCE_DATA = os.getenv(
    "REFERENCE_DATA", "data/processed/taxi_demand_features.parquet"
)
REFERENCE_PREDICTIONS = os.getenv(
    "REFERENCE_PREDICTIONS", "data/reference_predictions.csv"
)

OBJECT_NAME = os.getenv("DATA_DEST_PATH", "data/")
BUCKET = os.getenv("DATA_BUCKET", "data-files")

DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "10"))
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.20"))

PERFORMANCE_WINDOW_SIZE = int(os.getenv("PERFORMANCE_WINDOW_SIZE", "10"))
MAE_THRESHOLD = float(os.getenv("MAE_THRESHOLD", "50"))
RMSE_THRESHOLD = float(os.getenv("RMSE_THRESHOLD", "75"))

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

CSV_COLUMNS = ["request_id", "timestamp", "features", "prediction", "actual"]


def upload_to_minio(file_path):
    path = Path(file_path)
    if not path.exists():
        return

    dest_object = f"{OBJECT_NAME.rstrip('/')}/{path.name}"

    if upload_file is not None:
        try:
            upload_file(str(path), BUCKET, dest_object)
            print(f"[MINIO] Synced {file_path} -> s3://{BUCKET}/{dest_object}", flush=True)
            return
        except Exception as error:
            print(f"[MINIO] Helper upload failed, using boto3 fallback: {error}", flush=True)

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        )
        s3.upload_file(str(path), BUCKET, dest_object)
        print(f"[MINIO] Boto3 uploaded {file_path} -> s3://{BUCKET}/{dest_object}", flush=True)
    except Exception as error:
        print(f"[MINIO] ERROR uploading {file_path}: {error}", flush=True)


def download_from_minio(minio_key, local_path):
    dest = Path(local_path)
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        )
        s3.download_file(BUCKET, minio_key, str(dest))
        print(f"[MINIO] Downloaded s3://{BUCKET}/{minio_key} -> {dest}", flush=True)
        return True
    except Exception as error:
        print(f"[MINIO] Could not fetch s3://{BUCKET}/{minio_key}: {error}", flush=True)
        return False


def initialize_csv():
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. NEW: Try to restore existing state from MinIO first
    dest_object = f"{OBJECT_NAME.rstrip('/')}/{output_path.name}"
    restored = download_from_minio(dest_object, str(output_path))
    
    if restored and output_path.exists():
        print(f"[CSV] Restored historical state from MinIO: {dest_object}", flush=True)
        return

    # 2. Only create a blank file if no history exists in MinIO
    if not output_path.exists():
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        print(f"[CSV] Created fresh baseline file: {OUTPUT_FILE}", flush=True)
        upload_to_minio(OUTPUT_FILE)


def append_inference(request_id, timestamp, features, prediction):
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writerow(
            {
                "request_id": request_id,
                "timestamp": timestamp,
                "features": json.dumps(features),
                "prediction": prediction,
                "actual": "",
            }
        )
    print(f"[CSV] Saved inference: {request_id}", flush=True)
    upload_to_minio(OUTPUT_FILE)


def update_actual(request_id, actual):
    if not Path(OUTPUT_FILE).exists():
        return

    rows = []
    found = False

    with open(OUTPUT_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["request_id"] == request_id:
                row["actual"] = actual
                found = True
            rows.append(row)

    if not found:
        print(f"[CSV] WARNING: request_id not found: {request_id}", flush=True)
        return

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[CSV] Updated actual={actual} for request_id={request_id}",
        flush=True,
    )
    upload_to_minio(OUTPUT_FILE)


def load_reference_data():
    download_from_minio("data/processed/taxi_demand_features.parquet", REFERENCE_DATA)

    path = Path(REFERENCE_DATA)
    if not path.exists():
        print(f"[DRIFT] ERROR: Reference feature file unavailable at {path}", flush=True)
        return []

    try:
        if str(path).endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        rows = df.to_dict(orient="records")
        print(f"[DRIFT] Loaded {len(rows)} reference records from {path}", flush=True)
        return rows
    except Exception as error:
        print(f"[DRIFT] ERROR reading reference dataset {path}: {error}", flush=True)
        return []


def load_reference_predictions():
    download_from_minio("dummy/reference_predictions.csv", REFERENCE_PREDICTIONS)

    path = Path(REFERENCE_PREDICTIONS)
    if not path.exists():
        return []

    predictions = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                val = row.get("prediction")
                if val not in (None, ""):
                    predictions.append(float(val))
        print(f"[DRIFT] Loaded {len(predictions)} reference predictions from {path}", flush=True)
        return predictions
    except Exception as error:
        print(f"[DRIFT] ERROR reading reference predictions: {error}", flush=True)
        return []


def calculate_psi(reference_values, current_values, bins=5):
    if not reference_values or not current_values:
        return 0.0

    ref = [float(v) for v in reference_values]
    cur = [float(v) for v in current_values]

    minimum, maximum = min(ref), max(ref)
    if minimum == maximum:
        return 0.0

    width = (maximum - minimum) / bins
    ref_counts, cur_counts = [0] * bins, [0] * bins

    for v in ref:
        idx = max(0, min(int((v - minimum) / width), bins - 1))
        ref_counts[idx] += 1

    for v in cur:
        idx = max(0, min(int((v - minimum) / width), bins - 1))
        cur_counts[idx] += 1

    psi = 0.0
    for i in range(bins):
        ref_ratio = max(ref_counts[i] / len(ref), 0.0001)
        cur_ratio = max(cur_counts[i] / len(cur), 0.0001)
        psi += (cur_ratio - ref_ratio) * math.log(cur_ratio / ref_ratio)

    return psi


def detect_feature_drift(reference_rows, current_events):
    if not reference_rows:
        return False

    drift_detected = False
    for feature in FEATURE_COLUMNS:
        ref_vals = [
            float(r[feature])
            for r in reference_rows
            if feature in r and r[feature] is not None
        ]

        cur_vals = []
        for event in current_events:
            feats = event.get("features", {})
            if feature in feats and feats[feature] is not None:
                cur_vals.append(float(feats[feature]))

        if not ref_vals or not cur_vals:
            continue

        psi = calculate_psi(ref_vals, cur_vals)
        print(f"[DRIFT] Feature '{feature}': PSI={psi:.4f}", flush=True)
        if psi >= PSI_THRESHOLD:
            print(f"[ALERT] FEATURE DRIFT detected in column: {feature} (PSI={psi:.4f})", flush=True)
            drift_detected = True

    return drift_detected


def detect_prediction_drift(reference_predictions, current_events):
    if not reference_predictions or not current_events:
        return False

    cur_preds = [
        float(event["prediction"])
        for event in current_events
        if event.get("prediction") is not None
    ]

    if not cur_preds:
        return False

    psi = calculate_psi(reference_predictions, cur_preds)
    print(f"[DRIFT] Output Prediction PSI: {psi:.4f}", flush=True)

    if psi >= PSI_THRESHOLD:
        print(f"[ALERT] PREDICTION DRIFT detected! PSI ({psi:.4f}) >= Threshold ({PSI_THRESHOLD})", flush=True)
        return True

    return False


def detect_performance_drift():
    if not Path(OUTPUT_FILE).exists():
        return

    try:
        df = pd.read_csv(OUTPUT_FILE)
        labeled = df[df["actual"].notna() & (df["actual"] != "")].copy()

        if len(labeled) < PERFORMANCE_WINDOW_SIZE:
            return

        recent = labeled.tail(PERFORMANCE_WINDOW_SIZE)
        y_true = recent["actual"].astype(float)
        y_pred = recent["prediction"].astype(float)

        mae = (y_true - y_pred).abs().mean()
        rmse = math.sqrt(((y_true - y_pred) ** 2).mean())

        print(f"[PERFORMANCE] Window Evaluation ({len(recent)} samples): MAE={mae:.4f}, RMSE={rmse:.4f}", flush=True)

        if mae >= MAE_THRESHOLD:
            print(f"[ALERT] PERFORMANCE DRIFT detected! MAE ({mae:.4f}) >= Threshold ({MAE_THRESHOLD})", flush=True)

        if rmse >= RMSE_THRESHOLD:
            print(f"[ALERT] PERFORMANCE DRIFT detected! RMSE ({rmse:.4f}) >= Threshold ({RMSE_THRESHOLD})", flush=True)

    except Exception as error:
        print(f"[PERFORMANCE] Error calculating performance metrics: {error}", flush=True)


def create_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=KAFKA_GROUP_ID,
    )


def main():
    initialize_csv()
    reference_rows = load_reference_data()
    reference_predictions = load_reference_predictions()
    consumer = create_consumer()

    print("[MONITOR] Kafka listener active...", flush=True)
    current_window = []

    for message in consumer:
        try:
            event = message.value
            event_type = event.get("event_type")

            if event_type == "inference":
                request_id = event.get("request_id")
                if not request_id:
                    continue

                append_inference(
                    request_id,
                    event.get("timestamp"),
                    event.get("features", {}),
                    event.get("prediction"),
                )
                current_window.append(event)

                if len(current_window) >= DRIFT_WINDOW_SIZE:
                    detect_feature_drift(reference_rows, current_window)
                    detect_prediction_drift(reference_predictions, current_window)
                    current_window = []

            elif event_type == "feedback":
                update_actual(event.get("request_id"), event.get("actual"))
                detect_performance_drift()

        except Exception as error:
            print(f"[ERROR] Processing Kafka event failed: {error}", flush=True)


if __name__ == "__main__":
    main()