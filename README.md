# Real-Time Taxi Demand Forecasting and MLOps Deployment System

An end-to-end production MLOps system for real-time NYC taxi demand forecasting. This project covers the entire ML lifecycle: automated data processing and feature engineering, dataset versioning with DVC on MinIO S3 storage, multi-model experimentation and registry with MLflow, streaming ingestion with Apache Kafka, orchestration with Apache Airflow, and containerized deployment on Kubernetes with Prometheus telemetry and continuous data drift monitoring.

---

## Architecture Overview

                                \[ NYC TLC Taxi Data (Parquet) \]

                                               │

                                               ▼

                      ┌─────────────────────────────────────────────────┐

                      │  Data Engineering & Feature Pipeline (DVC)      │

                      │  \- scripts/process\_data.py                      │

                      │  \- Lags (1h, 2h, 24h), rolling stats, holidays  │

                      │  \- S3 / MinIO Object Storage (data-files)       │

                      └────────────────────────┬────────────────────────┘

                                               │

                                               ▼

                      ┌─────────────────────────────────────────────────┐

                      │  Model Training & MLflow Tracking (K8s Job)     │

                      │  \- k8s/pipeline-2-training.yaml                 │

                      │  \- Models: Baseline RF, XGBoost, LightGBM       │

                      │  \- Hyperparameter tuning via Optuna             │

                      │  \- PostgreSQL backend \+ MLflow Model Registry   │

                      └────────────────────────┬────────────────────────┘

                                               │

                                               ▼

                      ┌─────────────────────────────────────────────────┐

                      │  Serving & Real-Time Streaming (Kubernetes)     │

                      │  \- FastAPI REST Server (app/main.py :8000)      │

                      │  \- Apache Kafka ('taxi-events' stream :9092)    │

                      │  \- Apache Airflow DAG Orchestration (:8080)     │

                      └────────────────────────┬────────────────────────┘

                                               │

                                               ▼

                      ┌─────────────────────────────────────────────────┐

                      │  Production Telemetry & Drift Monitoring        │

                      │  \- Prometheus metrics scraper (:9090)           │

                      │  \- Drift Monitor: KS-test & Wasserstein drift   │

                      │  \- Simulated Label Producer for ground truth    │

                      └─────────────────────────────────────────────────┘

---

## Repository Layout

.

├── .dvcignore                      \# DVC ignore configuration

├── .gitignore                      \# Git ignore rules (.venv, data/, .env, cache)

├── data.dvc                        \# DVC dataset tracker (hash & metadata)

├── Dockerfile                      \# Multi-stage container for FastAPI & workers

├── Dockerfile.mlflow               \# Custom MLflow container with PostgreSQL driver

├── manual\_deployment.sh            \# Automated end-to-end Kubernetes deployment script

├── requirements.txt                \# Python environment dependencies

├── README.md                       \# Project documentation

├── app/

│   ├── main.py                     \# FastAPI application entrypoint

│   ├── schemas.py                  \# Pydantic input/output schemas

│   └── metrics.py                  \# Prometheus request & latency instrumentation

├── scripts/

│   ├── process\_data.py             \# Data cleaning & spatio-temporal feature engineering

│   ├── upload\_file.py              \# S3/MinIO bucket upload helper

│   └── evaluate\_model.py           \# Model evaluation & residual calculations

├── k8s/

│   ├── minio.yaml                  \# MinIO S3 storage deployment

│   ├── minio-init-job.yaml         \# MinIO bucket ('data-files') initialization job


│   ├── mlflow.yaml                 \# MLflow tracking server deployment

│   ├── kafka.yaml                  \# Apache Kafka broker deployment

│   ├── pipeline-2-training.yaml    \# Kubernetes Job for model training

│   ├── airflow.yaml                \# Apache Airflow webserver & scheduler

│   ├── deployment.yaml             \# FastAPI app deployment (taxi-api)

│   ├── service.yaml                \# FastAPI service configuration

│   ├── prometheus.yaml             \# Prometheus metrics server deployment

│   ├── drift-monitor.yaml          \# Real-time data & prediction drift monitor

│   └── label-producer.yaml         \# Simulated ground truth streaming job

└── tests/

    ├── test\_data\_pipeline.py       \# Data validation & feature schema tests

    └── test\_model\_training.py      \# Metric computation & inference sanity tests

---

## Core System Pipelines

### 1\. Data Engineering Pipeline (Pipeline 1\)

* **Ingestion & Filtering:** Ingests NYC Yellow Taxi parquet files (`yellow_tripdata_2024-01.parquet`), removes zero-passenger rides, invalid coordinates, negative durations, and outliers.  
    
* **Feature Engineering:**  
  * **Spatio-Temporal Aggregation:** Groups trip records into hourly pick-up demand buckets per `PULocationID`.  
  * **Calendar & Cyclical Signals:** Hour of day, day of week, weekend flags, public holidays (via `holidays` package), and cyclical $\\sin / \\cos$ hour transformations.  
  * **Autoregressive Lags:** Lagged demand features at $t-1$, $t-2$, and $t-24$ hours.  
  * **Moving Statistics:** 3-hour rolling mean demand.  
      
* **Storage & Versioning:** Seeds processed feature sets to MinIO S3 bucket `data-files` and tracks datasets with DVC (`data.dvc`).

### 

### 2\. Model Development Pipeline (Pipeline 2\)

* **Training Workflow:** Executed as an isolated Kubernetes Job (`k8s/pipeline-2-training.yaml`).  
    
* **Chronological Split:** 80% train / 20% validation split sorted chronologically to avoid future lookahead bias.  
    
* **Model Benchmarking:** Compares `RandomForestRegressor`, `XGBoostRegressor`, and `LightGBMRegressor`.  
    
* **Hyperparameter Optimization:** Automated Bayesian optimization across trees, learning rate, and depth using **Optuna**.  
    
* **MLflow Tracking & Registry:** Tracks runs, parameters, metrics (RMSE, MAE, $R^2$), and registers the top model (`TaxiDemandModel`) in `Production` stage with a PostgreSQL backend.


### 3\. Serving, Streaming & Orchestration (Pipeline 3\)

* **FastAPI Service:** High-performance REST API running via Uvicorn (`app.main:app`) exposing `/predict`, `/health`, and `/metrics`.  
    
* **Apache Kafka Streaming:** A dedicated Kafka broker (`kafka:9092`) with the `taxi-events` topic handles live trip event streaming.  
    
* **Airflow Orchestration:** Scheduled DAGs orchestrate periodic batch feature extraction and model validation.


### 4\. Telemetry & Drift Monitoring (Pipeline 4\)

* **Prometheus:** Scrapes request rates, HTTP status codes, and latency histograms from FastAPI on port 8000\.  
    
* **Continuous Drift Monitoring (`k8s/drift-monitor.yaml`):** Consumes live inference requests, pairs them with delayed labels from `simulated-label-producer`, and runs statistical two-sample tests (Kolmogorov-Smirnov and Wasserstein distance) against the training baseline to detect distribution shift.

---

## Quickstart & Deployment Guide

### Prerequisites

* Docker & Docker Compose  
* Local Kubernetes cluster (Docker Desktop K8s, Minikube, or Kind)  
* `kubectl` CLI tool  
* Python 3.11+

---

### **Method A:** Automated One-Click Deployment (Recommended)

1. Open your terminal in the target parent folder (where git is initialized).
2. Run: git clone -b integration/newfb --single-branch https://github.com/Darakhasha/RealTime-Taxi-Demand-Forecasting.git
3. Navigate into the cloned repository: cd RealTime-Taxi-Demand-Forecasting
4. Open Docker Desktop and ensure Kubernetes is enabled.
5. Execute the deployment script: ./manual_deployment.sh
6. If step 5 fails with a permission error, grant execution rights first by running:
    chmod +x manual_deployment.sh && ./manual_deployment.sh

This script will automatically:

1. Build the unified `taxi-api:latest` Docker image.  
2. Configure Kubernetes secrets for MinIO and PostgreSQL.  
3. Deploy MinIO and initialize the `data-files` storage bucket.  
4. Process and seed the raw and engineered feature data into MinIO via DVC.  
5. Deploy PostgreSQL, MLflow tracking server, and Kafka broker.  
6. Create the `taxi-events` Kafka topic.  
7. Trigger the `pipeline-2-training` Kubernetes Job and wait for model registration.  
8. Deploy Airflow, the FastAPI service, Prometheus, and the drift monitor.  
9. Establish background port-forwards for all web interfaces.

---

### **Method B:** Manual Step-by-Step Deployment

#### 1\. Local Environment Setup & Tests

python \-m venv .venv

source .venv/bin/activate  \# On Windows: .venv\\Scripts\\Activate.ps1

pip install \--upgrade pip

pip install \-r requirements.txt

\# Run unit tests

pytest tests/ \-v

#### 2\. Build Docker Images

docker build \-t taxi-api:latest .

docker build \-t mlflow-server:v2.10.2 \-f Dockerfile.mlflow .

#### 3\. Create Secrets & Deploy Storage

\# MinIO credentials secret

kubectl create secret generic minio-credentials \\

  \--from-literal=AWS\_ACCESS\_KEY\_ID="minioadmin" \\

  \--from-literal=AWS\_SECRET\_ACCESS\_KEY="minioadmin" \\

  \--from-literal=MINIO\_ENDPOINT="http://minio.default.svc.cluster.local:9000"

\# PostgreSQL credentials secret

kubectl create secret generic postgres-credentials \\

  \--from-literal=POSTGRES\_USER="airflow" \\

  \--from-literal=POSTGRES\_PASSWORD="airflow" \\

  \--from-literal=POSTGRES\_DB="airflow"

\# Deploy storage and database

kubectl apply \-f k8s/minio.yaml

kubectl apply \-f k8s/minio-init-job.yaml

kubectl apply \-f k8s/postgres.yaml

#### 4\. Seed MinIO with Processed Data

python scripts/process\_data.py \\

  \--input data/raw/yellow\_tripdata\_2024-01.parquet \\

  \--output data/processed/taxi\_demand\_features.parquet \\

  \--sample 0.2

python scripts/upload\_file.py \\

  \--bucket data-files \\

  \--filepath data/processed/taxi\_demand\_features.parquet \\

  \--dest-path data/processed/taxi\_demand\_features.parquet

#### 5\. Deploy Core Infrastructure & Train Model

kubectl apply \-f k8s/mlflow.yaml

kubectl apply \-f k8s/kafka.yaml

\# Create Kafka topic

kubectl exec deployment/kafka \-- /opt/kafka/bin/kafka-topics.sh \\

  \--create \--if-not-exists \--topic taxi-events \--bootstrap-server kafka:9092 \--partitions 1 \--replication-factor 1

\# Launch Model Training Kubernetes Job

kubectl apply \-f k8s/pipeline-2-training.yaml

#### 6\. Deploy Serving, Airflow & Monitoring

kubectl apply \-f k8s/airflow.yaml

kubectl apply \-f k8s/deployment.yaml

kubectl apply \-f k8s/service.yaml

kubectl apply \-f k8s/prometheus.yaml

kubectl apply \-f k8s/drift-monitor.yaml

kubectl apply \-f k8s/label-producer.yaml

---

## Service Endpoints & Web Consoles

After running the deployment script, all services are accessible locally via port-forwarding:

kubectl port-forward service/taxi-api-service 8000:8000 &

kubectl port-forward service/mlflow 5000:5000 &

kubectl port-forward service/minio 9001:9001 &

kubectl port-forward service/airflow-service 8080:8080 &

kubectl port-forward service/prometheus 9090:9090 &

| Service | Local URL | Default Credentials | Description |
| :---- | :---- | :---- | :---- |
| **Taxi Demand API** | `http://localhost:8000` | *None* | FastAPI Real-time Prediction Engine |
| **Interactive API Docs** | `http://localhost:8000/docs` | *None* | Swagger UI Documentation & Testing |
| **MLflow Registry** | `http://localhost:5000` | *None* | Experiment Tracking & Model Artifacts |
| **MinIO Console** | `http://localhost:9001` | `minioadmin` / `minioadmin` | S3 Storage Browser (`data-files`) |
| **Apache Airflow** | `http://localhost:8080` | `admin` / `admin` | DAG Pipeline Orchestrator |
| **Prometheus** | `http://localhost:9090` | *None* | Live Latency & Throughput Metrics |

---

## API Usage & Examples

### 1\. Real-Time Demand Prediction (`POST /predict`)

#### Request:

curl \-X POST "http://localhost:8000/predict" \\

     \-H "Content-Type: application/json" \\

     \-d '{

       "PULocationID": 132,

       "avg\_passenger\_count": 1.5,

       "avg\_trip\_distance": 2.85,

       "hour\_of\_day": 18,

       "day\_of\_week": 4,

       "is\_weekend": 0,

       "sin\_hour": \-0.9659,

       "cos\_hour": \-0.2588,

       "lag\_1h\_demand": 45.0,

       "lag\_2h\_demand": 38.0,

       "lag\_24h\_demand": 50.0,

       "rolling\_mean\_3h": 41.5

     }'

#### Response:

{

  "status": "success",

  "model\_name": "TaxiDemandModel",

  "model\_version": "1",

  "predicted\_demand": 46.28,

  "unit": "rides\_per\_hour"

}

### 2\. Service Health Check (`GET /health`)

curl \-X GET "http://localhost:8000/health"

{

  "status": "healthy",

  "model\_loaded": true,

  "stage": "Production"

}

### 3\. Prometheus Metrics Endpoint (`GET /metrics`)

curl \-X GET "http://localhost:8000/metrics"

---

## Teardown

To shut down and clean up all Kubernetes deployments, jobs, and port-forwarding processes:

kubectl delete \-f k8s/

pkill \-f "kubectl port-forward"  
