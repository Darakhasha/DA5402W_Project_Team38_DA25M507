import json
import os
import random
import time

from kafka import KafkaConsumer
from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "taxi-events",
)

LABEL_DELAY_SECONDS = float(
    os.getenv(
        "LABEL_DELAY_SECONDS",
        "2",
    )
)


consumer = KafkaConsumer(
    KAFKA_TOPIC,

    bootstrap_servers=(
        KAFKA_BOOTSTRAP_SERVERS
    ),

    value_deserializer=lambda value:
        json.loads(
            value.decode("utf-8")
        ),

    auto_offset_reset="latest",

    enable_auto_commit=True,

    group_id="demo-label-producer",
)


producer = KafkaProducer(
    bootstrap_servers=(
        KAFKA_BOOTSTRAP_SERVERS
    ),

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8"),
)


print(
    "Simulated label producer started."
)

print(
    "Waiting for inference events..."
)


for message in consumer:

    event = message.value

    # Ignore feedback events.
    if event.get("event_type") != "inference":
        continue

    request_id = event.get(
        "request_id"
    )

    prediction = event.get(
        "prediction"
    )

    if request_id is None:
        continue

    if prediction is None:
        continue

    print(
        f"Received prediction: "
        f"{prediction}"
    )

    # Simulate the delay between prediction
    # and availability of the actual label.
    time.sleep(
        LABEL_DELAY_SECONDS
    )

    # --------------------------------------------------
    # DEMO ONLY
    #
    # Simulate the actual taxi demand.
    #
    # Random error is added around the prediction.
    # --------------------------------------------------

    actual = prediction + random.randint(
        -15,
        15,
    )

    feedback_event = {
        "event_type": "feedback",
        "request_id": request_id,
        "actual": actual,
    }

    producer.send(
        KAFKA_TOPIC,
        value=feedback_event,
    ).get(timeout=10)

    print(
        f"Sent simulated actual: "
        f"{actual} "
        f"for request_id={request_id}"
    )