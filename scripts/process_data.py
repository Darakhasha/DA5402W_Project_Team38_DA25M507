import argparse
import os
import sys
import numpy as np
import pandas as pd


def clean_and_engineer(input_path: str, output_path: str):
    print(f"Loading raw dataset from {input_path}...")

    # 1. Load Data (Supports both CSV and Parquet formats)
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        # Load first 500k rows if dealing with massive raw CSVs for fast local execution
        df = pd.read_csv(input_path, nrows=500000)

    # 2. Identify datetime column dynamically and clean bad rows
    datetime_col = None
    for col in [
        'tpep_pickup_datetime',
        'pickup_datetime',
        'Trip_Pickup_DateTime',
    ]:
        if col in df.columns:
            datetime_col = col
            break

    if not datetime_col:
        raise ValueError('Could not find pickup timestamp column in dataset.')

    # Convert pickup times to Pandas datetimes
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Filter valid distances and fares (clean out anomalies on the fly)
    if 'trip_distance' in df.columns:
        df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
    if 'fare_amount' in df.columns:
        df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]

    df = df.dropna(subset=[datetime_col])

    # 3. Zone / Location ID Mapping
    if 'PULocationID' in df.columns:
        df['zone_id'] = df['PULocationID']
    elif 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
        # Fallback to spatial bin grid if raw coordinates are used instead of location zones
        df = df[
            (df['pickup_latitude'].between(40.5, 41.0))
            & (df['pickup_longitude'].between(-74.3, -73.7))
        ]
        df['zone_id'] = (
            df['pickup_latitude'].round(2).astype(str)
            + '_'
            + df['pickup_longitude'].round(2).astype(str)
        )
    else:
        df['zone_id'] = 1

    # 4. Hourly Spatial-Temporal Aggregation
    # Round down timestamps to the nearest hour (e.g., 17:23 -> 17:00)
    df['pickup_hour'] = df[datetime_col].dt.floor('h')

    agg_dict = {'demand': (datetime_col, 'count')}
    if 'trip_distance' in df.columns:
        agg_dict['avg_trip_distance'] = ('trip_distance', 'mean')
    if 'fare_amount' in df.columns:
        agg_dict['avg_fare_amount'] = ('fare_amount', 'mean')

    demand_df = (
        df.groupby(['pickup_hour', 'zone_id']).agg(**agg_dict).reset_index()
    )

    # 5. Extract Feature Attributes (Temporal Features)
    demand_df['hour_of_day'] = demand_df['pickup_hour'].dt.hour
    demand_df['day_of_week'] = demand_df['pickup_hour'].dt.dayofweek
    demand_df['is_weekend'] = (
        demand_df['day_of_week'].isin([4, 5]).astype(int)
    )

    # 6. Save Processed Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    demand_df.to_parquet(output_path, index=False)
    print(f'Processed dataset saved successfully to: {output_path}')


if __name__ == '__main__':
    # CHANGED: Replaced brittle sys.argv lookups and nested main blocks with argparse
    parser = argparse.ArgumentParser(
        description='Clean and engineer taxi demand features.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/yellow_tripdata_2024-01.parquet',
        help='Path to input raw dataset',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/taxi_demand_features.parquet',
        help='Path to save output parquet dataset',
    )

    args = parser.parse_args()
    clean_and_engineer(args.input, args.output)