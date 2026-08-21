import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# Default operational parameters for DAG tasks
default_args = {
    'owner': 'Zeba',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    # Resiliency: Task automatically retries 3 times if it fails, waiting 1 minute in between [20]
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

# Define the workflow container scheduling to execute every 5 minutes [20]
with DAG(
    'zeba_data_engineering_pipeline',
    default_args=default_args,
    schedule_interval='@daily', #'*/5 * * * *', # Cron expression for 'Every 5 minutes'
    catchup=False
) as dag:

    # Task 1: Ingest raw streaming data through Kafka [19]
    task_ingest = BashOperator(
        task_id='Ingest_Raw_Data',
        bash_command='python src/producer.py',
        priority_weight=10 # Assign a high priority weight to ingestion tasks [20]
    )

    # Helper Python function to validate that raw files are present on disk
    def validate_ingested_file():
        # Verifies if the dataset exist inside our project raw folder
        # if os.path.exists('data/yellow_tripdata_2024-01.parquet'):
        #     return 'processing_group.DataPreprocessing'
        # else:
        #     return 'Data_Validation_Failed'

        s3_hook = S3Hook(aws_conn_id='minio_default') # Ensure you set this connection in Airflow UI
        file_exists = s3_hook.check_for_key(
            key='data/yellow_tripdata_2024-01.parquet',
            bucket_name='data-files'
        )
        if file_exists:
            return 'processing_group.DataPreprocessing'
        else:
            return 'Data_Validation_Failed'

    # Task 2: Conditional Branching to validate data prior to feature engineering [19, 20]
    task_validate = BranchPythonOperator(
        task_id='DataValidation',
        python_callable=validate_ingested_file
    )

    # Task 2a: Handling validation failure
    task_failed_validation = BashOperator(
        task_id='Data_Validation_Failed',
        bash_command='echo "CRITICAL: Raw taxi dataset is missing from data/ directory! Aborting pipeline."'
    )

    # Task Group to organize Spark Preprocessing and Feature Engineering Tasks [20]
    with TaskGroup('processing_group') as processing_group:
        
        task_preprocess = BashOperator(
            task_id='DataPreprocessing',
            bash_command='echo "Executing Spark stream cleaning and formatting..."'
        )

        task_feature_eng = BashOperator(
            task_id='FeatureEngineering',
            bash_command='python src/streaming_pipeline.py'
        )

        # Sequence inside the TaskGroup
        task_preprocess >> task_feature_eng

    # Task 3: Store and Version Processed Features via DVC [5]
    task_dvc_version = BashOperator(
        task_id='Store_Processed_Dataset',
        bash_command='dvc add data/taxi_features_clean.parquet && dvc push'
    )

    # Define DAG workflow dependencies [20, 21]
    task_ingest >> task_validate
    task_validate >> processing_group >> task_dvc_version
    task_validate >> task_failed_validation