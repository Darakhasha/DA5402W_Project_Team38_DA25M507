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
    schedule_interval='*/5 * * * *',#'@daily',
    catchup=False,
) as dag:

    # 1. Pull raw Parquet dataset from MinIO via DVC
    task_dvc_pull = BashOperator(
        task_id='dvc_pull_raw_data',
        bash_command='''
            cd /opt/airflow && \
            mkdir -p data/raw data/processed && \
            dvc pull data/raw/yellow_tripdata_2024-01.parquet.dvc || true
        ''',
        env=dvc_env,
    )

    # 2. Clean raw data and engineer features
    task_clean_features = BashOperator(
        task_id='clean_and_engineer_features',
        bash_command='''
            cd /opt/airflow && \
            mkdir -p data/processed && \
            python scripts/process_data.py --input data/raw/yellow_tripdata_2024-01.parquet --output data/processed/taxi_demand_features.parquet
        ''',
    )



   # 3. Run Spark batch aggregation on Kafka stream
    task_spark_batch = BashOperator(
        task_id='run_spark_batch_aggregation',
        bash_command='cd /opt/airflow && python3 -m pip install --user duckdb && python src/streaming_pipeline.py',
    )


    # 4. Version final processed dataset with DVC and push to MinIO
    task_dvc_version = BashOperator(
        task_id='dvc_version_dataset',
        bash_command='cd /opt/airflow && dvc add data/processed/taxi_demand_features.parquet && dvc push',
        env=dvc_env,
    )

    task_dvc_pull >> task_clean_features >> task_spark_batch >> task_dvc_version