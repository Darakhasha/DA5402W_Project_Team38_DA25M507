import csv
import json
import os
from pathlib import Path

from kafka import KafkaConsumer


# ============================================================
# Configuration
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


# ============================================================
# CSV storage
# ============================================================

CSV_COLUMNS = [
    "request_id",
    "timestamp",
    "features",
    "prediction",
    "actual",
]


def initialize_csv():
    """
    Create the CSV file and directory if they do not exist.
    """

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
            f"Created monitoring file: "
            f"{OUTPUT_FILE}", flush=True
        )


def append_inference(
    request_id,
    timestamp,
    features,
    prediction,
):
    """
    Save one inference observation.

    Actual is initially empty because the true
    ground-truth value may not be available yet.
    """

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
                "features": json.dumps(
                    features
                ),
                "prediction": prediction,
                "actual": "",
            }
        )

    print(
        f"[CSV] Saved inference: "
        f"{request_id}"
    )


def update_actual(
    request_id,
    actual,
):
    """
    Find the inference row using request_id
    and update its actual value.
    """

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
            f"[CSV] WARNING: No inference found "
            f"for request_id={request_id}", flush=True
        )

        return

    # Rewrite CSV with updated actual value.
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
        f"for request_id={request_id}", flush=True
    )


# ============================================================
# Kafka
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

        auto_offset_reset= "earliest",#"latest",

        enable_auto_commit=True,

        group_id=KAFKA_GROUP_ID,
    )


# ============================================================
# Main monitoring loop
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "     INFERENCE DATA MONITOR STARTED"
    )

    print(
        "======================================"
    )

    print(
        f"Kafka: "
        f"{KAFKA_BOOTSTRAP_SERVERS}"
    )

    print(
        f"Topic: "
        f"{KAFKA_TOPIC}"
    )

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )

    print(
        "======================================"
    )

    initialize_csv()

    consumer = create_consumer()

    print(
        "Waiting for Kafka events..."
    )

    for message in consumer:

        try:

            event = message.value

            event_type = event.get(
                "event_type"
            )

            # ==================================================
            # Inference event
            # ==================================================

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
                        "[ERROR] "
                        "Inference event has no "
                        "request_id" , flush=True
                    )

                    continue

                append_inference(
                    request_id=request_id,
                    timestamp=timestamp,
                    features=features,
                    prediction=prediction,
                )

            # ==================================================
            # Feedback event
            # ==================================================

            elif event_type == "feedback":

                request_id = event.get(
                    "request_id"
                )

                actual = event.get(
                    "actual"
                )

                if not request_id:

                    print(
                        "[ERROR] "
                        "Feedback event has no "
                        "request_id", flush=True
                    )

                    continue

                if actual is None:

                    print(
                        "[ERROR] "
                        "Feedback event has no "
                        "actual value", flush=True
                    )

                    continue

                update_actual(
                    request_id=request_id,
                    actual=actual,
                )

            # ==================================================
            # Unknown event
            # ==================================================

            else:

                print(
                    f"[WARNING] Unknown "
                    f"event_type={event_type}", flush=True
                )

        except Exception as error:

            print(
                "[ERROR] Failed to process "
                f"Kafka event: {error}", flush=True
            )


if __name__ == "__main__":
    main()