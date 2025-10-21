"""
conftest.py

Shared pytest fixtures for testing
"""

import os

import mlflow
import pandas as pd
import pytest

from backend.utils import load_model_and_preprocessor
from src.core.utils import load_environment

# Load env and configurations
env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)
MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """Automatically cleanup MLflow runs before and after each test."""
    if mlflow.active_run():
        mlflow.end_run()
    yield
    if mlflow.active_run():
        mlflow.end_run()


@pytest.fixture(scope="module")
def model_and_preprocessor():
    """Load both model and preprocessor from MLflow or fallback to dummy model."""
    MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS")
    return load_model_and_preprocessor(MODEL_NAME, MODEL_ALIAS)


@pytest.fixture(scope="module")
def model(model_and_preprocessor):
    """Extract just the model."""
    model, _ = model_and_preprocessor
    return model


@pytest.fixture(scope="module")
def preprocessor(model_and_preprocessor):
    """Extract just the preprocessor."""
    _, preprocessor = model_and_preprocessor
    return preprocessor


@pytest.fixture
def sample_input():
    """Return a default sample input dictionary for API requests in raw PaySim format."""
    return {
        "step": 1,
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 181.0,
        "type": "CASH_OUT",
        "nameOrig": "C84071102",
        "nameDest": "C1576697216",
    }


@pytest.fixture
def mock_raw_df():
    """Return a larger sample DataFrame for testing pipeline components
    with enough data for SMOTE."""
    # Create 50 samples to ensure enough data after train/test split for SMOTE
    n_samples = 50
    fraud_pattern = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]  # 30% fraud rate
    fraud_labels = (fraud_pattern * (n_samples // len(fraud_pattern) + 1))[:n_samples]

    return pd.DataFrame(
        {
            "step": list(range(1, n_samples + 1)),
            "amount": [100.0 + i * 50 for i in range(n_samples)],
            "oldbalanceOrg": [1000.0 + i * 100 for i in range(n_samples)],
            "newbalanceOrig": [900.0 + i * 90 for i in range(n_samples)],
            "oldbalanceDest": [i * 25 for i in range(n_samples)],
            "newbalanceDest": [100.0 + i * 30 for i in range(n_samples)],
            "type": ["CASH_OUT", "TRANSFER", "CASH_IN", "PAYMENT", "DEBIT"]
            * (n_samples // 5),
            "nameOrig": [f"C{i:03d}" for i in range(n_samples)],
            "nameDest": [f"M{i:03d}" for i in range(n_samples)],
            "isFlaggedFraud": [0] * n_samples,
            "isFraud": fraud_labels,
        }
    )


@pytest.fixture
def model_server_ip():
    """Return the model server IP for integration tests."""
    return MODEL_SERVER_IP
