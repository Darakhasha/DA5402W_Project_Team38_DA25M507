from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'Zeba',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'taxi_demand_data_engineering_pipeline',
    default_args=default_args,
    description='Automated pipeline for taxi demand data cleaning, feature engineering, and DVC tracking',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # 1. Clean and engineer features
    task_clean_features = BashOperator(
        task_id='clean_and_engineer_features',
        bash_command='python scripts/process_data.py data/raw/yellow_tripdata_2015-01.csv data/processed/taxi_demand_features.parquet'
    )

    # 2. Version processed dataset with DVC
    task_dvc_version = BashOperator(
        task_id='dvc_version_dataset',
        bash_command='dvc add data/processed/taxi_demand_features.parquet && dvc push'
    )

    task_clean_features >> task_dvc_version