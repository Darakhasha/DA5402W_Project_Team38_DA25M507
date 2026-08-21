"""
Pipeline 3 - Champion model loader.

Pipeline 2 is responsible for:
    train -> evaluate -> select best -> register/promote champion in MLflow.

Pipeline 3 is responsible for:
    load the champion model -> serve predictions.

The champion is identified using an MLflow Model Registry alias.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import mlflow


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "TaxiDemandModel",
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)


@dataclass
class LoadedModel:
    model: Any
    model_name: str
    model_version: str
    model_uri: str


class ModelNotLoadedError(RuntimeError):
    pass


_lock = threading.Lock()
_loaded_model: Optional[LoadedModel] = None


def _load_from_mlflow() -> Optional[LoadedModel]:
    """
    Load the model currently assigned to the champion alias.
    Returns None gracefully if the model is not found yet.
    """

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        model = mlflow.pyfunc.load_model(model_uri)

        # Get model version information from the registry.
        client = mlflow.MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI
        )

        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        champion_version = None

        for version in versions:
            if version.aliases and MODEL_ALIAS in version.aliases:
                champion_version = version.version
                break

        if champion_version is None:
            champion_version = "unknown"

        return LoadedModel(
            model=model,
            model_name=MODEL_NAME,
            model_version=str(champion_version),
            model_uri=model_uri,
        )

    except Exception as exc:
        # Gracefully log instead of crashing on startup
        print(f"Warning: Could not load champion model '{MODEL_NAME}@{MODEL_ALIAS}' yet: {exc}")
        return None


def get_model() -> Optional[LoadedModel]:
    """
    Return the cached champion model.

    If not loaded yet, attempts to load it without raising a fatal error.
    """

    global _loaded_model

    if _loaded_model is None:

        with _lock:

            if _loaded_model is None:
                _loaded_model = _load_from_mlflow()

    return _loaded_model


def reload_model() -> Optional[LoadedModel]:
    """
    Force Pipeline 3 to reload the current MLflow champion.

    Useful when Pipeline 2 promotes a new model version.
    """

    global _loaded_model

    with _lock:
        _loaded_model = _load_from_mlflow()

    return _loaded_model