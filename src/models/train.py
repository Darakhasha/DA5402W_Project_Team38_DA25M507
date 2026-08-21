import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def setup_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "Taxi_Demand_Forecasting_Comparison"
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location="s3://models/taxi_demand_experiments"
        )
    mlflow.set_experiment(experiment_name)

def load_data():
    local_path = "data/processed/taxi_demand_features.parquet"
    
    # Try MinIO/S3 first, fallback to local path
    try:
        print("Loading parquet files from MinIO...")
        df = pd.read_parquet(
            "s3://data-files/processed/",
            storage_options={
                "key": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
                "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
                "client_kwargs": {"endpoint_url": os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")}
            }
        )
    except Exception as exc:
        print(f"MinIO read failed ({exc}). Loading local file {local_path}...")
        df = pd.read_parquet(local_path)
    
    # Auto-repair missing feature columns if needed
    if 'hour_of_day' in df.columns:
        if 'sin_hour' not in df.columns:
            df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
        if 'cos_hour' not in df.columns:
            df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)

    feature_cols = [
        "PULocationID", "avg_trip_distance", "avg_fare_amount",
        "hour_of_day", "day_of_week", "is_weekend", "sin_hour", "cos_hour"
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    target_col = "demand"
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = np.ascontiguousarray(train_df[feature_cols].values, dtype=np.float32)
    y_train = np.ascontiguousarray(train_df[target_col].values, dtype=np.float32)
    X_test = np.ascontiguousarray(test_df[feature_cols].values, dtype=np.float32)
    y_test = np.ascontiguousarray(test_df[target_col].values, dtype=np.float32)
    
    return X_train, X_test, y_train, y_test

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_compare_models():
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data()
    
    models = {
        "Baseline_RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost_Regressor": XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
    }
    
    results = {}
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"--- Training {model_name} ---")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            rmse, mae, r2 = eval_metrics(y_test, predictions)
            results[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            
            mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            if "XGBoost" in model_name:
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
                
            print(f"{model_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    train_and_compare_models()