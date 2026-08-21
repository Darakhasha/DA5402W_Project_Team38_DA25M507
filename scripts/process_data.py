import argparse
import os
import numpy as np
import pandas as pd

def clean_and_engineer(input_path: str, output_path: str):
    print(f"Loading raw dataset from {input_path}...")

    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, nrows=500000)

    # Primary pickup timestamp column from official schema
    datetime_col = 'tpep_pickup_datetime' if 'tpep_pickup_datetime' in df.columns else None
    if not datetime_col:
        for col in ['pickup_datetime', 'Trip_Pickup_DateTime']:
            if col in df.columns:
                datetime_col = col
                break

    if not datetime_col:
        raise ValueError('Could not find pickup timestamp column in dataset.')

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    if 'trip_distance' in df.columns:
        df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
    if 'fare_amount' in df.columns:
        df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]

    df = df.dropna(subset=[datetime_col])

    if 'PULocationID' in df.columns:
        df['PULocationID'] = pd.to_numeric(df['PULocationID'], errors='coerce').fillna(1).astype(int)
    else:
        df['PULocationID'] = 1

    df['pickup_hour'] = df[datetime_col].dt.floor('h')

    agg_dict = {'demand': (datetime_col, 'count')}
    if 'trip_distance' in df.columns:
        agg_dict['avg_trip_distance'] = ('trip_distance', 'mean')
    if 'fare_amount' in df.columns:
        agg_dict['avg_fare_amount'] = ('fare_amount', 'mean')

    demand_df = df.groupby(['pickup_hour', 'PULocationID']).agg(**agg_dict).reset_index()

    # Feature Engineering
    demand_df['hour_of_day'] = demand_df['pickup_hour'].dt.hour
    demand_df['day_of_week'] = demand_df['pickup_hour'].dt.dayofweek
    demand_df['is_weekend'] = demand_df['day_of_week'].isin([5, 6]).astype(int)
    demand_df['sin_hour'] = np.sin(2 * np.pi * demand_df['hour_of_day'] / 24.0)
    demand_df['cos_hour'] = np.cos(2 * np.pi * demand_df['hour_of_day'] / 24.0)

    demand_df = demand_df.drop(columns=['pickup_hour'])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    demand_df.to_parquet(output_path, index=False)
    print(f'Processed dataset saved successfully to: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/yellow_tripdata_2024-01.parquet')
    parser.add_argument('--output', type=str, default='data/processed/taxi_demand_features.parquet')
    args = parser.parse_args()
    clean_and_engineer(args.input, args.output)