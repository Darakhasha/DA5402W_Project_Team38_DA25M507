import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def load_data():
    df = pd.read_parquet("data/processed/taxi_demand_features.parquet")
    
    feature_cols = [
        "PULocationID", "avg_passenger_count", "avg_trip_distance",
        "hour_of_day", "day_of_week", "is_weekend", "sin_hour", "cos_hour",
        "lag_1h_demand", "lag_2h_demand", "lag_24h_demand", "rolling_mean_3h"
    ]
    target_col = "demand"
    
    # 80% train / 20% test temporal split
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
    mlflow.set_experiment("Taxi_Demand_Forecasting_Comparison")
    X_train, X_test, y_train, y_test = load_data()
    
    # Compare Baseline vs Advanced Gradient Boosted Trees
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

    print("\nModel Comparison Summary:")
    print(pd.DataFrame(results).T)

if __name__ == "__main__":
    train_and_compare_models()