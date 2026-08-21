from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

dvc_env = {
    'AWS_ACCESS_KEY_ID': 'minioadmin',
    'AWS_SECRET_ACCESS_KEY': 'minioadmin',
    'AWS_DEFAULT_REGION': 'us-east-1',
}

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

    # 1. Pull raw data from MinIO using the git-tracked .dvc file
    task_dvc_pull = BashOperator(
        task_id='dvc_pull_raw_data',
        bash_command='dvc pull data/raw/yellow_tripdata_2015-01.csv.dvc',
        env=dvc_env,
    )

    # 2. Clean and engineer features
    task_clean_features = BashOperator(
        task_id='clean_and_engineer_features',
        bash_command='python scripts/process_data.py --input data/raw/yellow_tripdata_2015-01.csv --output data/processed/taxi_demand_features.parquet',
    )

    # 3. Run Spark batch aggregation
    task_spark_batch = BashOperator(
        task_id='run_spark_batch_aggregation',
        bash_command='python scripts/streaming_pipeline.py',
    )

    # 4. Version processed dataset with DVC and push to MinIO
    task_dvc_version = BashOperator(
        task_id='dvc_version_dataset',
        bash_command='dvc add data/processed/taxi_demand_features.parquet && dvc push',
        env=dvc_env,
    )

    task_dvc_pull >> task_clean_features >> task_spark_batch >> task_dvc_version