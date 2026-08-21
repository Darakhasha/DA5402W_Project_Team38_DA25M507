import json
import os
import random
import time
from kafka import KafkaConsumer, KafkaProducer

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "taxi-events")
LABEL_DELAY_SECONDS = float(os.getenv("LABEL_DELAY_SECONDS", "2"))

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="demo-label-producer",
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

print("Simulated label producer started.", flush=True)
print("Waiting for inference events...\n", flush=True)

for message in consumer:
    event = message.value

    if event.get("event_type") != "inference":
        continue

    request_id = event.get("request_id")
    prediction = event.get("prediction")
    features = event.get("features", {})

    if request_id is None or prediction is None:
        continue

    drift_mode = event.get("drift_mode") or features.get("drift_mode", "normal")

    print(f"Received prediction: {prediction} | Mode: {drift_mode}", flush=True)

    time.sleep(LABEL_DELAY_SECONDS)

    if drift_mode in ["performance_drift", "all_drift"]:
        actual = prediction + random.randint(80, 120)
    else:
        actual = prediction + random.randint(-5, 5)

    feedback_event = {
        "event_type": "feedback",
        "request_id": request_id,
        "actual": actual,
    }

    producer.send(KAFKA_TOPIC, value=feedback_event).get(timeout=10)

    print(f"Sent simulated actual: {actual} for request_id={request_id}\n", flush=True)