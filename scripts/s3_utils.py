import os
import boto3

# Dynamically read endpoint from environment, falling back to 127.0.0.1 for host runner
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
print(MINIO_ENDPOINT)
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def upload_file(local_path, bucket, object_name):
    # Auto-create bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=bucket)
    except Exception:
        s3_client.create_bucket(Bucket=bucket)
    
    print(f"Uploading {local_path} to {bucket}/{object_name}...")
    s3_client.upload_file(local_path, bucket, object_name)
    print("Upload successful!")

def download_file(bucket, object_name, local_path):
    print(f"Downloading {bucket}/{object_name} to {local_path}...")
    s3_client.download_file(bucket, object_name, local_path)