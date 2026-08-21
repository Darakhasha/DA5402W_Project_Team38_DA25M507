import os
import sys
import optuna
import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Add current directory to path for clean imports
sys.path.append(os.path.dirname(__file__))
# CHANGED: Added setup_mlflow to the import list
from train import load_data, setup_mlflow

def objective(trial):
    X_train, X_test, y_train, y_test = load_data()
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42
    }
    
    with mlflow.start_run(nested=True):
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        mlflow.log_params(params)
        mlflow.log_metric("val_rmse", rmse)
        
    return rmse

def run_tuning(n_trials: int = 10):
    # CHANGED: Initialize MLflow with MinIO and SQLite settings before running
    setup_mlflow()
    
    mlflow.set_experiment("Taxi_Demand_Hyperparameter_Tuning")
    with mlflow.start_run(run_name="Optuna_XGBoost_Study"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        print(f"\nBest Optimization Trial RMSE: {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_rmse", study.best_value)

if __name__ == "__main__":
    run_tuning()