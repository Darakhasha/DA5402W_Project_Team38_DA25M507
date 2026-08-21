"""
Structured logging for Darshita's deployment pipeline.

- app_logger  -> human-readable service logs (startup, errors, model reloads) -> logs/app.log
- pred_logger -> one JSON line per prediction request/response -> logs/predictions.log

NOTE: Monitoring (Pipeline 4) operates via Kafka streams (topic: taxi-events) 
rather than tailing these log files directly. The local files are retained purely 
for local debugging and fallback auditability.
"""
from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

APP_LOG_PATH = os.path.join(LOG_DIR, "app.log")
PRED_LOG_PATH = os.path.join(LOG_DIR, "predictions.log")


def _build_app_logger() -> logging.Logger:
    logger = logging.getLogger("darshita.app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler = RotatingFileHandler(APP_LOG_PATH, maxBytes=2_000_000, backupCount=3)
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def _build_pred_logger() -> logging.Logger:
    logger = logging.getLogger("darshita.predictions")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(PRED_LOG_PATH, maxBytes=5_000_000, backupCount=5)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


app_logger = _build_app_logger()
pred_logger = _build_pred_logger()


def log_prediction(record: dict) -> None:
    """Append one JSON line per prediction, so downstream monitoring can tail the file."""
    pred_logger.info(json.dumps(record, default=str))
