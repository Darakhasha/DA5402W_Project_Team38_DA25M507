import json
import os

from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "taxi-events",
)

_producer = None


def get_producer():
    global _producer
    if os.getenv("TESTING") == "true":
        return None
        
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value).encode("utf-8")
        )
    return _producer


def publish_inference(
    request_id: str,
    timestamp: str,
    features: dict,
    prediction,
) -> None:

    producer = get_producer()
    if producer is None:
        return
    
    event = {
        "event_type": "inference",
        "request_id": request_id,
        "timestamp": timestamp,
        "features": features,
        "prediction": prediction,
    }

    producer.send(
        KAFKA_TOPIC,
        value=event,
    ).get(timeout=10)


def publish_feedback(
    request_id: str,
    actual,
) -> None:

    producer = get_producer()
    if producer is None:
        return
    
    event = {
        "event_type": "feedback",
        "request_id": request_id,
        "actual": actual,
    }

    producer.send(
        KAFKA_TOPIC,
        value=event,
    ).get(timeout=10)