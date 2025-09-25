"""
conftest.py

Shared pytest fixtures for FYP tests:
- model: loads MLflow model once per module.
- sample_input: provides default feature input for API tests.
- model_server_ip: returns model server URL for integration/e2e tests.

Ensure MLflow and model server are running when using model or model_server_ip.
"""
import os
import mlflow

import pytest
from dotenv import load_dotenv
from src.utils import load_config
from api.main import load_model

load_dotenv()
config = load_config()
api_config = config.get("api", {})
MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@pytest.fixture(scope="module")
def model():
    """Load the ML model once for all tests."""
    mdl = load_model()
    assert mdl is not None
    return mdl

@pytest.fixture
def sample_input():
    """Return a default sample input dictionary for API requests."""
    feature_names = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
    return {"features": {name: 0.0 for name in feature_names}}

@pytest.fixture
def model_server_ip():
    """Return the model server URL."""
    return MODEL_SERVER_IP
