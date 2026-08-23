#!/bin/bash
set -e

echo "========================================"
echo "Starting Local MLOps Manual Deployment"
echo "========================================"

IMAGE_NAME="taxi-api"
IMAGE_TAG="latest"
MLFLOW_IMAGE="ghcr.io/darakhasha/mlflow:latest"

echo "1. Building Docker Images..."
docker build -f Dockerfile.mlflow -t ${MLFLOW_IMAGE} .
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "2. Deploying MinIO Storage & Buckets..."
kubectl apply -f k8s/minio.yaml
kubectl rollout status deployment/minio --timeout=120s

kubectl exec deployment/minio -- mc alias set myminio http://localhost:9000 minioadmin minioadmin
kubectl exec deployment/minio -- mc mb --ignore-existing myminio/models

kubectl apply -f k8s/minio-init-job.yaml
kubectl wait --for=condition=complete job/minio-init-job --timeout=60s

echo "3. Seeding MinIO with Processed Data..."
kubectl port-forward deployment/minio 9000:9000 > /dev/null 2>&1 &
PF_PID=$!
sleep 5 

export MINIO_ENDPOINT="http://127.0.0.1:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"

dvc remote modify minio-store endpointurl "http://127.0.0.1:9000"
dvc remote modify --local minio-store access_key_id minioadmin
dvc remote modify --local minio-store secret_access_key minioadmin
dvc pull || echo "Warning: DVC pull skipped or failed because remote storage is empty."

python scripts/process_data.py --input data/raw/yellow_tripdata_2024-01.parquet --output data/processed/taxi_demand_features.parquet
python scripts/upload_file.py --bucket data-files --filepath data/processed/taxi_demand_features.parquet --dest-path data/processed/taxi_demand_features.parquet

kill $PF_PID

echo "4. Deploying Ingestion Pipeline..."
kubectl apply -f k8s/pipeline-1-ingestion.yaml

echo "5. Cleaning State & Deploying Postgres Metastore..."
kubectl delete deployment airflow-postgres --ignore-not-found=true
kubectl delete svc airflow-postgres --ignore-not-found=true
kubectl delete deployment airflow --ignore-not-found=true
kubectl delete deployment mlflow --ignore-not-found=true

kubectl apply -f k8s/postgres.yaml
echo "Waiting for PostgreSQL engine..."
until kubectl exec deployment/airflow-postgres -- pg_isready -U airflow -d airflow > /dev/null 2>&1; do
  sleep 2
done

echo "Preparing MLflow and Airflow schemas..."
chk=$(kubectl exec deployment/airflow-postgres -- psql -U airflow -d airflow -tAc "SELECT 1 FROM pg_database WHERE datname='mlflow'")
if [[ "$chk" != *"1"* ]]; then
  kubectl exec deployment/airflow-postgres -- psql -U airflow -d airflow -c "CREATE DATABASE mlflow;"
fi
kubectl exec deployment/airflow-postgres -- psql -U airflow -d airflow -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

echo "6. Deploying MLflow & Model Training Job..."
kubectl apply -f k8s/mlflow.yaml
kubectl rollout restart deployment/mlflow
kubectl rollout status deployment/mlflow --timeout=180s

kubectl apply -f k8s/pipeline-2-training.yaml

echo "7. Deploying Airflow & Provisioning Dependencies..."
kubectl create configmap airflow-dags --from-file=dags/ --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap airflow-src --from-file=src/ --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap airflow-scripts --from-file=scripts/ --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/airflow.yaml
kubectl rollout status deployment/airflow --timeout=300s

AIRFLOW_POD=$(kubectl get pod -l app=airflow -o jsonpath='{.items[0].metadata.name}')
kubectl exec $AIRFLOW_POD -c airflow -- mkdir -p /opt/airflow/data/raw
if [ -f "data/raw/yellow_tripdata_2024-01.parquet" ]; then
  kubectl cp data/raw/yellow_tripdata_2024-01.parquet $AIRFLOW_POD:/opt/airflow/data/raw/yellow_tripdata_2024-01.parquet -c airflow
fi

kubectl exec $AIRFLOW_POD -c airflow -- bash -c "export PIP_USER=false && python3 -m pip install duckdb pyspark"
kubectl exec $AIRFLOW_POD -c airflow -- airflow dags reserialize

echo "8. Deploying API Workloads..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/taxi-api --timeout=120s

python scripts/smoke_test.py

echo "9. Deploying Kafka & Prometheus..."
kubectl apply -f k8s/prometheus.yaml
kubectl rollout status deployment/prometheus --timeout=120s

kubectl apply -f k8s/kafka.yaml
kubectl rollout status deployment/kafka --timeout=180s
kubectl exec deployment/kafka -- /opt/kafka/bin/kafka-topics.sh --create --if-not-exists --topic taxi-events --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1

echo "10. Deploying Monitoring Stack..."
kubectl apply -f k8s/drift-monitor.yaml
kubectl apply -f k8s/label-producer.yaml
kubectl rollout status deployment/drift-monitor --timeout=300s
kubectl rollout status deployment/simulated-label-producer --timeout=300s

echo "========================================"
echo "Establishing Background Port-Forwards..."
echo "========================================"
kubectl port-forward service/taxi-api-service 8000:8000 > /dev/null 2>&1 &
kubectl port-forward deployment/minio 9000:9000 > /dev/null 2>&1 &
kubectl port-forward deployment/minio 9001:9001 > /dev/null 2>&1 &
kubectl port-forward deployment/kafka 9092:9092 > /dev/null 2>&1 &
kubectl port-forward deployment/mlflow 5000:5000 > /dev/null 2>&1 &
kubectl port-forward deployment/prometheus 9090:9090 > /dev/null 2>&1 &
kubectl port-forward deployment/airflow 8080:8080 > /dev/null 2>&1 &

echo "========================================"
echo "Manual Deployment Complete!"
echo "API: http://localhost:8000"
echo "MLflow: http://localhost:5000"
echo "MinIO: http://localhost:9001 (minioadmin / minioadmin)"
echo "Airflow: http://localhost:8080 (admin / admin)"
echo "Prometheus: http://localhost:9090"
echo "========================================"