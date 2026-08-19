import os
import boto3

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio.default.svc.cluster.local:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def ensure_bucket_exists(bucket_name: str):
    s3 = get_s3_client()
    existing_buckets = [b['Name'] for b in s3.list_buckets().get('Buckets', [])]
    if bucket_name not in existing_buckets:
        s3.create_bucket(Bucket=bucket_name)

def upload_file(local_path: str, bucket: str, object_name: str):
    ensure_bucket_exists(bucket)
    s3 = get_s3_client()
    s3.upload_file(local_path, bucket, object_name)
    print(f"Uploaded {local_path} to s3://{bucket}/{object_name}")

def download_file(bucket: str, object_name: str, local_path: str):
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, object_name, local_path)
    print(f"Downloaded s3://{bucket}/{object_name} to {local_path}")