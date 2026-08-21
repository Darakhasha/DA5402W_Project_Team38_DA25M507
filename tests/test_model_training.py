import os
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

# Prevent Kafka connections from hanging test runs
os.environ["TESTING"] = "true"


# 1. Test evaluation metrics computation
def test_eval_metrics_computation():
    from src.models.train import eval_metrics

    actual = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([11.0, 19.0, 29.0, 42.0])

    rmse, mae, r2 = eval_metrics(actual, pred)
    assert rmse > 0.0, "RMSE must be positive"
    assert mae > 0.0, "MAE must be positive"
    assert r2 <= 1.0, "R2 score cannot exceed 1.0"


# 2. Test processed data loading
def test_data_loading():
    data_path = "data/processed/taxi_demand_features.parquet"
    if not os.path.exists(data_path):
        pytest.skip(f"Processed dataset {data_path} not found.")

    from src.models.train import load_data

    # Unpack the 4 data splits directly
    X_train, X_test, y_train, y_test = load_data()[:4]

    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Testing set is empty"

    # Safely convert to numpy arrays for NaN evaluation
    X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train_np = y_train.to_numpy() if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train

    assert not np.isnan(X_train_np).any(), "Found NaN values in training features"
    assert not np.isnan(y_train_np).any(), "Found NaN values in training targets"


# 3. Test model fitting and prediction sanity
def test_model_training_and_prediction():
    X_dummy = np.random.rand(50, 8).astype(np.float32)
    y_dummy = np.random.rand(50).astype(np.float32)

    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_dummy, y_dummy)
    preds = model.predict(X_dummy[:10])

    assert preds.shape == (10,), "Prediction output shape mismatch"
    assert not np.isnan(preds).any(), "Model produced NaN predictions"