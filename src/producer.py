import pandas as pd
import json
import time
from kafka import KafkaProducer
import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

# 1. Establish connection to your local Kafka Broker
print("Connecting to local Kafka broker...")
producer = KafkaProducer(
    bootstrap_servers= KAFKA_BOOTSTRAP_SERVERS, #['localhost:9092'],
    # Serialize our structured dictionary data to JSON and encode as binary bytes
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = 'taxi_demand_stream'
# Points to the official NYC Yellow Taxi dataset downloaded programmatically or manually
file_path = 'data/yellow_tripdata_2024-01.parquet'

print(f"Ingesting raw dataset from: {file_path}")
# Read the compressed, binary Parquet file directly into Pandas [2]
df = pd.read_parquet(file_path)

# Limit to 50,000 rows for local system testing to save processing memory [3]
df_sample = df.head(50000)

print("Starting real-time streaming ingestion...")
for index, row in df_sample.iterrows():
    # Construct the JSON payload with the exact field names expected by our pipeline [4]
    data = {
        "PULocationID": str(row["PULocationID"]),
        "tpep_pickup_datetime": str(row["tpep_pickup_datetime"]),
        "fare_amount": float(row["fare_amount"])
    }
    
    # Send the event payload to the Kafka topic
    producer.send(topic_name, value=data)
    print(f"Sent Event -> Zone: {data['PULocationID']}, Time: {data['tpep_pickup_datetime']}, Fare: ${data['fare_amount']:.2f}")
    
    # Introduce a 0.5-second sleep interval to simulate live incoming data [5]
    time.sleep(0.5)

# Ensure all messages are flushed to the broker before disconnecting
producer.flush()
producer.close()
print("Ingestion complete.")