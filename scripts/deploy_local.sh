#!/bin/bash
# scripts/deploy_local.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables (allows overriding from the command line)
IMAGE_NAME=${IMAGE_NAME:-"ghcr.io/your_github_username/taxi-api"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

echo "Deploying with Image: ${IMAGE_NAME}:${IMAGE_TAG}"

echo "=== Deploying MinIO ==="
kubectl apply -f k8s/minio-secret.yaml
kubectl apply -f k8s/minio.yaml
kubectl rollout status deployment/minio --timeout=120s

echo "=== Deploying Ingestion & Serving APIs ==="
kubectl apply -f k8s/pipeline-1-ingestion.yaml
kubectl set image cronjob/pipeline-1-ingestion ingestion-container=${IMAGE_NAME}:${IMAGE_TAG}

kubectl apply -f k8s/pipeline-3-serving.yaml
kubectl set image deployment/pipeline-3-serving serving-container=${IMAGE_NAME}:${IMAGE_TAG}
kubectl rollout status deployment/pipeline-3-serving --timeout=120s

echo "=== Deploying Core Taxi API ==="
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl set image deployment/taxi-api taxi-api=${IMAGE_NAME}:${IMAGE_TAG}
kubectl rollout status deployment/taxi-api --timeout=120s

echo "=== Deploying Prometheus ==="
kubectl apply -f k8s/prometheus.yaml
kubectl rollout restart deployment/prometheus
kubectl rollout status deployment/prometheus --timeout=120s

echo "=== Deploying Kafka ==="
kubectl apply -f k8s/kafka.yaml
kubectl rollout status deployment/kafka --timeout=180s
kubectl exec deployment/kafka -- /opt/kafka/bin/kafka-topics.sh --create --if-not-exists --topic taxi-events --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1

echo "=== Deploying Cloudflare Tunnels ==="
kubectl apply -f k8s/cloudflared.yaml
kubectl rollout status deployment/cloudflare-tunnels --timeout=60s

echo "=== Deploying Monitors & Producers ==="
kubectl apply -f k8s/drift-monitor.yaml
kubectl set image deployment/drift-monitor drift-monitor=${IMAGE_NAME}:${IMAGE_TAG}

kubectl apply -f k8s/label-producer.yaml
kubectl set image deployment/simulated-label-producer simulated-label-producer=${IMAGE_NAME}:${IMAGE_TAG}

kubectl rollout status deployment/drift-monitor --timeout=300s
kubectl rollout status deployment/simulated-label-producer --timeout=300s

echo "=== Local Deployment Complete! ==="