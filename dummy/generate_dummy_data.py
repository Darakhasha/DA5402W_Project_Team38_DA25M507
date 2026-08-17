import numpy as np
import pandas as pd


np.random.seed(42)


# ============================================================
# 1. TRAINING / REFERENCE DATA
# ============================================================

n_training = 1000

timestamps = pd.date_range(
    start="2026-01-01",
    periods=n_training,
    freq="h"
)

hour = timestamps.hour
day_of_week = timestamps.dayofweek

temperature = np.random.normal(
    loc=28,
    scale=4,
    size=n_training
)

rain = np.random.binomial(
    n=1,
    p=0.2,
    size=n_training
)

traffic_index = np.random.uniform(
    low=20,
    high=80,
    size=n_training
)

taxi_demand = (
    50
    + 20 * ((hour >= 7) & (hour <= 9))
    + 25 * ((hour >= 17) & (hour <= 20))
    + 0.4 * traffic_index
    + 5 * rain
    + np.random.normal(0, 5, n_training)
)

taxi_demand = np.maximum(
    0,
    taxi_demand
).astype(int)


training_data = pd.DataFrame({
    "timestamp": timestamps,
    "hour": hour,
    "day_of_week": day_of_week,
    "temperature": temperature,
    "rain": rain,
    "traffic_index": traffic_index,
    "taxi_demand": taxi_demand
})


training_data.to_csv(
    "dummy/training_data.csv",
    index=False
)


# ============================================================
# 2. INCOMING / CURRENT DATA
# ============================================================

n_incoming = 300

incoming_timestamps = pd.date_range(
    start="2026-02-12",
    periods=n_incoming,
    freq="h"
)

incoming_hour = incoming_timestamps.hour
incoming_day_of_week = incoming_timestamps.dayofweek

incoming_temperature = np.random.normal(
    loc=29,
    scale=4,
    size=n_incoming
)

incoming_rain = np.random.binomial(
    n=1,
    p=0.2,
    size=n_incoming
)

incoming_traffic_index = np.random.uniform(
    low=20,
    high=80,
    size=n_incoming
)

incoming_taxi_demand = (
    50
    + 20 * ((incoming_hour >= 7) & (incoming_hour <= 9))
    + 25 * ((incoming_hour >= 17) & (incoming_hour <= 20))
    + 0.4 * incoming_traffic_index
    + 5 * incoming_rain
    + np.random.normal(0, 5, n_incoming)
)

incoming_taxi_demand = np.maximum(
    0,
    incoming_taxi_demand
).astype(int)


incoming_data = pd.DataFrame({
    "timestamp": incoming_timestamps,
    "hour": incoming_hour,
    "day_of_week": incoming_day_of_week,
    "temperature": incoming_temperature,
    "rain": incoming_rain,
    "traffic_index": incoming_traffic_index,
    "taxi_demand": incoming_taxi_demand
})


incoming_data.to_csv(
    "dummy/incoming_data.csv",
    index=False
)


print("Dummy data generation completed.")
print()
print("Training data:")
print(training_data.head())
print()
print("Incoming data:")
print(incoming_data.head())