import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(experiment_name="Taxi_Demand_Forecasting_Comparison", model_name="TaxiDemandModel"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found. Run train.py first.")
        
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
    
    # Transition to Production stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    print(f"Registered '{model_name}' Version {model_version.version} into 'Production'")

if __name__ == "__main__":
    register_best_model()