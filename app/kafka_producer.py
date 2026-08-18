import json
import os

from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092"
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "taxi-events"
)

_producer = None

# producer = KafkaProducer(
#     bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
#     value_serializer=lambda value: json.dumps(value).encode("utf-8")
# )

def get_producer():
    global _producer
    # Skip creating a real producer if running unit tests
    if os.getenv("TESTING") == "true":
        return None
        
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value).encode("utf-8")
        )
    return _producer

def publish_prediction_request(request_data: dict) -> None:
    """
    Publish the exact prediction-request data to Kafka.

    The function does not assume any particular feature names.
    """
    producer = get_producer()
    if producer is None:
        return
    
    future = producer.send(
        KAFKA_TOPIC,
        value=request_data
    )

    # Wait for Kafka acknowledgement.
    # This makes failures visible instead of silently ignoring them.
    future.get(timeout=10)