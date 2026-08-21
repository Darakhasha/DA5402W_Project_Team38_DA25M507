import mlflow
import os
import sys
from mlflow.tracking import MlflowClient
import requests


sys.path.append(os.path.dirname(__file__))
from train import setup_mlflow

def register_best_model(experiment_name="Taxi_Demand_Forecasting_Comparison", model_name="TaxiDemandModel"):
    setup_mlflow()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    
    if not runs:
        raise RuntimeError("No runs found in experiment.")
        
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_rmse = best_run.data.metrics.get("rmse")
    print(f"Selected Best Run ID: {best_run_id} (RMSE: {best_rmse:.4f})")
    
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    # Assign champion alias for Pipeline 3 serving API compatibility
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version.version
    )
    print(f"Successfully registered '{model_name}' Version {model_version.version} as '@champion'")

def notify_api_reload():
    # Use your Kubernetes internal service DNS name
    api_url = os.getenv("API_RELOAD_URL", "http://taxi-api-service:8000/model/reload")
    try:
        print(f"Notifying API to hot-reload model at {api_url}...")
        response = requests.post(api_url, timeout=15)
        if response.status_code == 200:
            print("API successfully hot-swapped the new champion model!")
        else:
            print(f"API reload returned status {response.status_code}: {response.text}")
    except Exception as exc:
        print(f"Could not reach API reload endpoint (API might still be starting): {exc}")

if __name__ == "__main__":
    register_best_model()

    notify_api_reload()