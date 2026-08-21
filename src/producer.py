import os
import json
import time
import pandas as pd
from kafka import KafkaProducer

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

print("Connecting to Kafka broker...")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = 'taxi_demand_stream'

file_path = 'data/raw/yellow_tripdata_2024-01.parquet'

if not os.path.exists(file_path):
    file_path = 'data/raw/yellow_tripdata_2015-01.csv'
if not os.path.exists(file_path):
    file_path = 'data/yellow_tripdata_2024-01.parquet'

print(f"Ingesting raw dataset from: {file_path}")

if file_path.endswith('.parquet'):
    df = pd.read_parquet(file_path)
else:
    df = pd.read_csv(file_path, nrows=50000)

df_sample = df.head(50000)

print("Starting streaming ingestion...")
for index, row in df_sample.iterrows():
    # Enforce integer type for PULocationID and float for numeric metrics
    pu_id = int(row["PULocationID"]) if "PULocationID" in row and pd.notnull(row["PULocationID"]) else 1
    pickup_dt = str(row.get("tpep_pickup_datetime", ""))
    fare = float(row.get("fare_amount", 0.0))
    trip_dist = float(row.get("trip_distance", 0.0))
    
    data = {
        "PULocationID": pu_id,
        "tpep_pickup_datetime": pickup_dt,
        "fare_amount": fare,
        "trip_distance": trip_dist
    }
    
    producer.send(topic_name, value=data)
    time.sleep(0.001)

producer.flush()
producer.close()
print("Ingestion complete.")