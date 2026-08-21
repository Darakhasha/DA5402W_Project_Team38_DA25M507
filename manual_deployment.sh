#!/bin/bash
set -e

echo "========================================"
echo "Starting Local MLOps Deployment"
echo "========================================"

IMAGE_NAME="taxi-api"
IMAGE_TAG="latest"

echo "1. Building the Unified Docker Image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "2. Creating Kubernetes Secrets..."
kubectl create secret generic minio-credentials \
  --from-literal=AWS_ACCESS_KEY_ID="minioadmin" \
  --from-literal=AWS_SECRET_ACCESS_KEY="minioadmin" \
  --from-literal=MINIO_ENDPOINT="http://minio.default.svc.cluster.local:9000" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic postgres-credentials \
  --from-literal=POSTGRES_USER="airflow" \
  --from-literal=POSTGRES_PASSWORD="airflow" \
  --from-literal=POSTGRES_DB="airflow" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "3. Deploying MinIO Storage..."
kubectl apply -f k8s/minio.yaml
kubectl rollout status deployment/minio --timeout=120s
kubectl apply -f k8s/minio-init-job.yaml
kubectl wait --for=condition=complete job/minio-init-job --timeout=60s

echo "4. Seeding MinIO with Processed Data..."
kubectl port-forward deployment/minio 9000:9000 > /dev/null 2>&1 &
PF_PID=$!
sleep 5 

export MINIO_ENDPOINT="http://127.0.0.1:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"

dvc remote add -d minio-store s3://data-files --force
dvc remote modify minio-store endpointurl "http://127.0.0.1:9000"
dvc remote modify --local minio-store access_key_id minioadmin
dvc remote modify --local minio-store secret_access_key minioadmin
dvc pull || echo "No DVC data to pull yet."
python scripts/process_data.py --input data/raw/yellow_tripdata_2024-01.parquet --output data/processed/taxi_demand_features.parquet --sample 0.2
python scripts/upload_file.py --bucket data-files --filepath data/processed/taxi_demand_features.parquet --dest-path data/processed/taxi_demand_features.parquet
python scripts/upload_file.py --bucket data-files --filepath data/raw/yellow_tripdata_2024-01.parquet --dest-path data/raw/yellow_tripdata_2024-01.parquet

kill $PF_PID

echo "5. Deploying Core Infrastructure (Postgres, MLflow, Kafka)..."
kubectl apply -f k8s/postgres.yaml
kubectl rollout status deployment/airflow-postgres --timeout=120s

kubectl apply -f k8s/mlflow.yaml
kubectl rollout status deployment/mlflow --timeout=120s

kubectl apply -f k8s/kafka.yaml
kubectl rollout status deployment/kafka --timeout=180s

echo "6. Creating Kafka Topic..."
kubectl exec deployment/kafka -- /opt/kafka/bin/kafka-topics.sh --create --if-not-exists --topic taxi-events --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1

echo "7. Triggering Model Training..."
kubectl apply -f k8s/pipeline-2-training.yaml
kubectl wait --for=condition=complete job/pipeline-2-training --timeout=300s

echo "8. Deploying Orchestration & API..."
kubectl apply -f k8s/airflow.yaml
kubectl rollout status deployment/airflow --timeout=180s

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/taxi-api --timeout=120s

echo "9. Deploying Monitoring Stack..."
kubectl apply -f k8s/prometheus.yaml
kubectl rollout status deployment/prometheus --timeout=120s

kubectl apply -f k8s/drift-monitor.yaml
kubectl rollout status deployment/drift-monitor --timeout=120s

kubectl apply -f k8s/label-producer.yaml
kubectl rollout status deployment/simulated-label-producer --timeout=120s

echo "========================================"
echo "Establishing Background Port-Forwards..."
echo "========================================"
kubectl port-forward service/taxi-api-service 8000:8000 > /dev/null 2>&1 &
kubectl port-forward service/minio 9001:9001 > /dev/null 2>&1 &
kubectl port-forward service/kafka 9092:9092 > /dev/null 2>&1 &
kubectl port-forward service/mlflow 5000:5000 > /dev/null 2>&1 &
kubectl port-forward service/prometheus 9090:9090 > /dev/null 2>&1 &
kubectl port-forward service/airflow-service 8080:8080 > /dev/null 2>&1 &

echo "========================================"
echo "Deployment Complete & Port-Forwarded!"
echo "API: http://localhost:8000"
echo "MLflow: http://localhost:5000"
echo "MinIO: http://localhost:9001 (minioadmin / minioadmin)"
echo "Airflow: http://localhost:8080 (admin / admin)"
echo "Prometheus: http://localhost:9090"
echo "========================================"