import os
import pytest
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# 1. Test evaluation metrics computation
def test_eval_metrics_computation():
    from src.models.train import eval_metrics
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([11.0, 19.0, 29.0, 42.0])
    
    rmse, mae, r2 = eval_metrics(actual, pred)
    assert rmse > 0.0, "RMSE must be positive"
    assert mae > 0.0, "MAE must be positive"
    assert 0.0 <= r2 <= 1.0, "R2 must be between 0 and 1"

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
    assert not np.isnan(X_train).any(), "Found NaN values in training features"
    assert not np.isnan(y_train).any(), "Found NaN values in training targets"

# 3. Test model fitting and prediction sanity
def test_model_training_and_prediction():
    X_dummy = np.random.rand(50, 12).astype(np.float32)
    y_dummy = np.random.rand(50).astype(np.float32)
    
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_dummy, y_dummy)
    preds = model.predict(X_dummy[:10])
    
    assert preds.shape == (10,), "Prediction output shape mismatch"
    assert not np.isnan(preds).any(), "Model produced NaN predictions"