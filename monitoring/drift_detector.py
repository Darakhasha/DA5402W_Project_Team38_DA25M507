import csv
import json
import math
import os 
import subprocess
import sys
from pathlib import Path

from kafka import KafkaConsumer
from s3_utils import upload_file 

# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "taxi-events",
)

KAFKA_GROUP_ID = os.getenv(
    "KAFKA_GROUP_ID",
    "taxi-monitoring",
)

OUTPUT_FILE = os.getenv(
    "INFERENCE_OUTPUT_FILE",
    "monitoring/data/inference_data.csv",
)

REFERENCE_DATA = os.getenv(
    "REFERENCE_DATA",
    "dummy/training_data.csv",
)

REFERENCE_PREDICTIONS = os.getenv(
    "REFERENCE_PREDICTIONS",
    "dummy/reference_predictions.csv",
)

LOCAL_PATH =  os.getenv(
                "DATA_LOCAL_PATH",
                "dummy/",
            )	

OBJECT_NAME =  os.getenv(
		"DATA_DEST_PATH",
		"data/",
	) 	

BUCKET =  os.getenv(
                        "DATA_BUCKET",
                        "data-files",
                    )

DRIFT_WINDOW_SIZE = int(
    os.getenv(
        "DRIFT_WINDOW_SIZE",
        "10",
    )
)

PSI_THRESHOLD = float(
    os.getenv(
        "PSI_THRESHOLD",
        "0.20",
    )
)

PERFORMANCE_WINDOW_SIZE = int(
    os.getenv(
        "PERFORMANCE_WINDOW_SIZE",
        "10",
    )
)

MAE_THRESHOLD = float(
    os.getenv(
        "MAE_THRESHOLD",
        "50",
    )
)

RMSE_THRESHOLD = float(
    os.getenv(
        "RMSE_THRESHOLD",
        "75",
    )
)
# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "temperature",
    "rain",
    "traffic_index",
]


# ============================================================
# CSV STORAGE
# ============================================================

CSV_COLUMNS = [
    "request_id",
    "timestamp",
    "features",
    "prediction",
    "actual",
]



def initialize_csv():

    output_path = Path(OUTPUT_FILE)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not output_path.exists():

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS,
            )

            writer.writeheader()

        print(
            f"[CSV] Created: {OUTPUT_FILE}",
            flush=True,
        )


def append_inference(
    request_id,
    timestamp,
    features,
    prediction,
):

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writerow(
            {
                "request_id": request_id,
                "timestamp": timestamp,
                "features": json.dumps(features),
                "prediction": prediction,
                "actual": "",
            }
        )

    print(
        f"[CSV] Saved inference: {request_id}",
        flush=True,
    )


def update_actual(
    request_id,
    actual,
):

    if not Path(OUTPUT_FILE).exists():
        return

    rows = []
    found = False

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["request_id"] == request_id:

                row["actual"] = actual
                found = True

            rows.append(row)

    if not found:

        print(
            f"[CSV] WARNING: request_id not found: "
            f"{request_id}",
            flush=True,
        )

        return

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[CSV] Updated actual={actual} "
        f"for request_id={request_id}",
        flush=True,
    )


# ============================================================
# REFERENCE DATA
# ============================================================

def load_reference_data():

    path = Path(REFERENCE_DATA)

    if not path.exists():

        print(
            f"[DRIFT] ERROR: Reference data not found: "
            f"{REFERENCE_DATA}",
            flush=True,
        )

        return []

    try:

        with open(
            path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            rows = list(reader)

        print(
            f"[DRIFT] Loaded {len(rows)} reference rows "
            f"from {REFERENCE_DATA}",
            flush=True,
        )

        return rows

    except Exception as error:

        print(
            f"[DRIFT] ERROR reading reference data: "
            f"{error}",
            flush=True,
        )

        return []


def load_reference_predictions():

    path = Path(REFERENCE_PREDICTIONS)

    if not path.exists():

        print(
            f"[DRIFT] WARNING: Reference predictions not found: "
            f"{REFERENCE_PREDICTIONS}",
            flush=True,
        )

        return []

    predictions = []

    try:

        with open(
            path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                value = row.get("prediction")

                if value in (None, ""):
                    continue

                try:

                    predictions.append(
                        float(value)
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    continue

        print(
            f"[DRIFT] Loaded "
            f"{len(predictions)} reference predictions",
            flush=True,
        )

        return predictions

    except Exception as error:

        print(
            f"[DRIFT] ERROR reading reference predictions: "
            f"{error}",
            flush=True,
        )

        return []


# ============================================================
# PSI
# ============================================================

def calculate_psi(
    reference_values,
    current_values,
    bins=10,
):

    if not reference_values or not current_values:
        return None

    reference = [
        float(value)
        for value in reference_values
    ]

    current = [
        float(value)
        for value in current_values
    ]

    minimum = min(reference)
    maximum = max(reference)

    if minimum == maximum:
        return 0.0

    width = (
        maximum - minimum
    ) / bins

    reference_counts = [0] * bins
    current_counts = [0] * bins

    for value in reference:

        index = int(
            (value - minimum) / width
        )

        index = max(
            0,
            min(index, bins - 1),
        )

        reference_counts[index] += 1

    for value in current:

        index = int(
            (value - minimum) / width
        )

        index = max(
            0,
            min(index, bins - 1),
        )

        current_counts[index] += 1

    reference_total = len(reference)
    current_total = len(current)

    psi = 0.0

    for i in range(bins):

        reference_ratio = (
            reference_counts[i]
            / reference_total
        )

        current_ratio = (
            current_counts[i]
            / current_total
        )

        # Prevent log(0)
        reference_ratio = max(
            reference_ratio,
            0.0001,
        )

        current_ratio = max(
            current_ratio,
            0.0001,
        )

        psi += (
            current_ratio - reference_ratio
        ) * math.log(
            current_ratio / reference_ratio
        )

    return psi


# ============================================================
# FEATURE DRIFT
# ============================================================

def detect_feature_drift(
    reference_rows,
    current_events,
):

    print(
        "",
        flush=True,
    )

    print(
        "========== FEATURE DRIFT ==========",
        flush=True,
    )

    if not reference_rows:

        print(
            "[DRIFT] Cannot calculate feature drift: "
            "reference data unavailable",
            flush=True,
        )

        return False

    drift_detected = False

    for feature in FEATURE_COLUMNS:

        reference_values = []
        current_values = []

        # -----------------------------
        # Reference distribution
        # -----------------------------

        for row in reference_rows:

            value = row.get(feature)

            if value in (None, ""):
                continue

            try:

                reference_values.append(
                    float(value)
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

        # -----------------------------
        # Current distribution
        # -----------------------------

        for event in current_events:

            features = event.get(
                "features",
                {},
            )

            value = features.get(feature)

            if value in (None, ""):
                continue

            try:

                current_values.append(
                    float(value)
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

        if not reference_values:

            print(
                f"{feature}: no reference data",
                flush=True,
            )

            continue

        if not current_values:

            print(
                f"{feature}: no current data",
                flush=True,
            )

            continue

        psi = calculate_psi(
            reference_values,
            current_values,
        )

        print(
            f"{feature}: PSI={psi:.4f}",
            flush=True,
        )

        if psi >= PSI_THRESHOLD:

            print(
                f"[ALERT] DRIFT detected in "
                f"{feature}",
                flush=True,
            )

            drift_detected = True

    if drift_detected:

        print(
            "!!! FEATURE DRIFT DETECTED !!!",
            flush=True,
        )

    else:

        print(
            "NO FEATURE DRIFT",
            flush=True,
        )

    return drift_detected


# ============================================================
# PREDICTION DRIFT
# ============================================================

def detect_prediction_drift(
    reference_predictions,
    current_events,
):

    print(
        "",
        flush=True,
    )

    print(
        "========== PREDICTION DRIFT ==========",
        flush=True,
    )

    if not reference_predictions:

        print(
            "[DRIFT] Prediction drift cannot be "
            "calculated because reference predictions "
            "are unavailable.",
            flush=True,
        )

        return False

    current_predictions = []

    for event in current_events:

        prediction = event.get(
            "prediction"
        )

        if prediction in (None, ""):
            continue

        try:

            current_predictions.append(
                float(prediction)
            )

        except (
            ValueError,
            TypeError,
        ):

            continue

    if not current_predictions:

        print(
            "[DRIFT] No current predictions.",
            flush=True,
        )

        return False

    psi = calculate_psi(
        reference_predictions,
        current_predictions,
    )

    print(
        f"Prediction PSI={psi:.4f}",
        flush=True,
    )

    if psi >= PSI_THRESHOLD:

        print(
            "!!! PREDICTION DRIFT DETECTED !!!",
            flush=True,
        )

        return True

    print(
        "NO PREDICTION DRIFT",
        flush=True,
    )

    return False


# ============================================================
# DRIFT WINDOW
# ============================================================

def run_drift_detection(
    reference_rows,
    reference_predictions,
    current_window,
):

    print(
        "",
        flush=True,
    )

    print(
        "########################################",
        flush=True,
    )

    print(
        "       RUNNING DRIFT DETECTION",
        flush=True,
    )

    print(
        f"Window size: {len(current_window)}",
        flush=True,
    )

    print(
        "########################################",
        flush=True,
    )

    feature_drift = detect_feature_drift(
        reference_rows,
        current_window,
    )

    prediction_drift = detect_prediction_drift(
        reference_predictions,
        current_window,
    )

    print(
        "",
        flush=True,
    )

    print(
        "=============== RESULT ===============",
        flush=True,
    )

    print(
        f"Feature drift    : {feature_drift}",
        flush=True,
    )

    print(
        f"Prediction drift : {prediction_drift}",
        flush=True,
    )

    if feature_drift or prediction_drift:

        print(
            "!!! DRIFT DETECTED !!!",
            flush=True,
        )

    else:

        print(
            "NO DRIFT DETECTED",
            flush=True,
        )

    print(
        "======================================",
        flush=True,
    )


# ============================================================
# KAFKA
# ============================================================

def create_consumer():

    return KafkaConsumer(

        KAFKA_TOPIC,

        bootstrap_servers=(
            KAFKA_BOOTSTRAP_SERVERS
        ),

        value_deserializer=lambda value:
            json.loads(
                value.decode("utf-8")
            ),

        auto_offset_reset="earliest",

        enable_auto_commit=True,

        group_id=KAFKA_GROUP_ID,
    )


# ============================================================
# PREDICTION PERFORMANCE
# ============================================================

def calculate_prediction_performance():

    if not Path(OUTPUT_FILE).exists():

        print(
            "[PERFORMANCE] Inference CSV not found",
            flush=True,
        )

        return

    predictions = []
    actuals = []

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            prediction = row.get("prediction")
            actual = row.get("actual")

            # Only use records for which
            # the real label has arrived.
            if prediction in (None, ""):
                continue

            if actual in (None, ""):
                continue

            try:

                predictions.append(
                    float(prediction)
                )

                actuals.append(
                    float(actual)
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

    if len(predictions) < PERFORMANCE_WINDOW_SIZE:

        print(
            f"[PERFORMANCE] Waiting for labels: "
            f"{len(predictions)}/"
            f"{PERFORMANCE_WINDOW_SIZE}",
            flush=True,
        )

        return

    # Use the most recent labeled observations
    predictions = predictions[
        -PERFORMANCE_WINDOW_SIZE:
    ]

    actuals = actuals[
        -PERFORMANCE_WINDOW_SIZE:
    ]

    errors = []

    squared_errors = []

    for prediction, actual in zip(
        predictions,
        actuals,
    ):

        error = prediction - actual

        errors.append(
            abs(error)
        )

        squared_errors.append(
            error ** 2
        )

    mae = (
        sum(errors)
        / len(errors)
    )

    rmse = math.sqrt(
        sum(squared_errors)
        / len(squared_errors)
    )

    print(
        "",
        flush=True,
    )

    print(
        "========== PREDICTION PERFORMANCE ==========",
        flush=True,
    )

    print(
        f"Labeled samples: {len(predictions)}",
        flush=True,
    )

    print(
        f"MAE:  {mae:.4f}",
        flush=True,
    )

    print(
        f"RMSE: {rmse:.4f}",
        flush=True,
    )

    print(
        f"MAE threshold:  {MAE_THRESHOLD}",
        flush=True,
    )

    print(
        f"RMSE threshold: {RMSE_THRESHOLD}",
        flush=True,
    )

    performance_degraded = (
        mae > MAE_THRESHOLD
        or rmse > RMSE_THRESHOLD
    )

    if performance_degraded:

        print(
            "!!! MODEL PERFORMANCE DEGRADED !!!",
            flush=True,
        )

    else:

        print(
            "MODEL PERFORMANCE OK",
            flush=True,
        )

    print(
        "=============================================",
        flush=True,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "degraded": performance_degraded,
    }
# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================",
        flush=True,
    )

    print(
        "     INFERENCE DATA MONITOR STARTED",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    print(
        f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}",
        flush=True,
    )

    print(
        f"Topic: {KAFKA_TOPIC}",
        flush=True,
    )

    print(
        f"Output: {OUTPUT_FILE}",
        flush=True,
    )

    print(
        f"Reference data: {REFERENCE_DATA}",
        flush=True,
    )

    print(
        f"Reference predictions: "
        f"{REFERENCE_PREDICTIONS}",
        flush=True,
    )

    print(
        f"Window size: {DRIFT_WINDOW_SIZE}",
        flush=True,
    )

    print(
        f"PSI threshold: {PSI_THRESHOLD}",
        flush=True,
    )

    print(
        "======================================",
        flush=True,
    )

    initialize_csv()

    reference_rows = load_reference_data()

    reference_predictions = (
        load_reference_predictions()
    )

    consumer = create_consumer()

    print(
        "Waiting for Kafka events...",
        flush=True,
    )

    current_window = []

    for message in consumer:

        try:

            event = message.value

            event_type = event.get(
                "event_type"
            )

            # ================================================
            # INFERENCE
            # ================================================

            if event_type == "inference":

                request_id = event.get(
                    "request_id"
                )

                timestamp = event.get(
                    "timestamp"
                )

                features = event.get(
                    "features",
                    {},
                )

                prediction = event.get(
                    "prediction"
                )

                if not request_id:

                    print(
                        "[ERROR] Inference event "
                        "has no request_id",
                        flush=True,
                    )

                    continue

                append_inference(
                    request_id=request_id,
                    timestamp=timestamp,
                    features=features,
                    prediction=prediction,
                )

                current_window.append(
                    event
                )

                print(
                    f"[KAFKA] Received inference "
                    f"{len(current_window)}/"
                    f"{DRIFT_WINDOW_SIZE}",
                    flush=True,
                )

                # ============================================
                # RUN AFTER 10 EVENTS
                # ============================================

                if len(current_window) >= DRIFT_WINDOW_SIZE:

                    run_drift_detection(
                        reference_rows,
                        reference_predictions,
                        current_window,
                    )

                    # Start next window
                    current_window = []

            # ================================================
            # FEEDBACK
            # ================================================

            elif event_type == "feedback":

                request_id = event.get(
                    "request_id"
                )

                actual = event.get(
                    "actual"
                )

                if not request_id:

                    print(
                        "[ERROR] Feedback event "
                        "has no request_id",
                        flush=True,
                    )

                    continue

                if actual is None:

                    print(
                        "[ERROR] Feedback event "
                        "has no actual value",
                        flush=True,
                    )

                    continue

                update_actual(
                    request_id=request_id,
                    actual=actual,
                )

                calculate_prediction_performance()

            # ================================================
            # UNKNOWN EVENT
            # ================================================

            else:

                print(
                    f"[WARNING] Unknown event_type="
                    f"{event_type}",
                    flush=True,
                )

            if os.path.exists(OUTPUT_FILE):
               subprocess.run([
                            sys.executable,
                            "scripts/upload_flie.py",
                            "--bucket", BUCKET,
                            "--object-name", f"{OBJECT_NAME}/{OUTPUT_FILE.replace("\\", "/").split("/")[-1] }",
                            "--file-path", f"{LOCAL_PATH}/{OUTPUT_FILE.replace("\\", "/").split("/")[-1] }"
                        ], check=True)

            if os.path.exists(REFERENCE_PREDICTIONS):
                        subprocess.run([
                                    sys.executable,
                                    "scripts/upload_flie.py",
                                    "--bucket", BUCKET,
                                    "--object-name", f"{OBJECT_NAME}/{REFERENCE_PREDICTIONS.replace("\\", "/").split("/")[-1] }",
                                    "--file-path", f"{LOCAL_PATH}/{REFERENCE_PREDICTIONS.replace("\\", "/").split("/")[-1] }"
                                ], check=True)
        except Exception as error:

            print(
                "[ERROR] Failed to process "
                f"Kafka event: {error}",
                flush=True,
            )


if __name__ == "__main__":
    main()