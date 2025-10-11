"""
conftest.py

Shared pytest fixtures for FYP tests:
- model: loads MLflow model once per module.
- sample_input: provides default feature input for API tests.
- model_server_ip: returns model server URL for integration/e2e tests.
- mock_raw_df: provides sample DataFrame for testing pipeline components.

Ensure MLflow and model server are running when using model or model_server_ip.
"""

import os

import mlflow
import pandas as pd
import pytest

from src.utils import load_environment

# Use ENV_FILE if set, otherwise default to .env
env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)
MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@pytest.fixture(scope="module")
def model():
    """Load the ML model once for all tests."""
    MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS")

    try:
        # Try to load model using alias first
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        mdl = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback to version 1 if alias doesn't exist
        model_version = 1
        model_uri = f"models:/{MODEL_NAME}/{model_version}"
        mdl = mlflow.sklearn.load_model(model_uri)

    assert mdl is not None
    return mdl


@pytest.fixture
def sample_input():
    """Return a default sample input dictionary for API requests."""
    return {
        "step": 1,
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "type": "CASH_OUT",
        "nameOrig_token": 123,
        "nameDest_token": 456,
    }


@pytest.fixture
def mock_raw_df():
    """Return a sample DataFrame for testing pipeline components."""
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "amount": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
                1100.0,
                1200.0,
            ],
            "oldbalanceOrg": [
                1000.0,
                2000.0,
                3000.0,
                4000.0,
                5000.0,
                6000.0,
                7000.0,
                8000.0,
                9000.0,
                10000.0,
                11000.0,
                12000.0,
            ],
            "newbalanceOrig": [
                900.0,
                1800.0,
                2700.0,
                3600.0,
                4500.0,
                5400.0,
                6300.0,
                7200.0,
                8100.0,
                9000.0,
                9900.0,
                10800.0,
            ],
            "oldbalanceDest": [
                0.0,
                500.0,
                1000.0,
                1500.0,
                2000.0,
                2500.0,
                3000.0,
                3500.0,
                4000.0,
                4500.0,
                5000.0,
                5500.0,
            ],
            "newbalanceDest": [
                100.0,
                700.0,
                1300.0,
                1900.0,
                2500.0,
                3100.0,
                3700.0,
                4300.0,
                4900.0,
                5500.0,
                6100.0,
                6700.0,
            ],
            "type": [
                "CASH_OUT",
                "TRANSFER",
                "CASH_IN",
                "PAYMENT",
                "CASH_OUT",
                "TRANSFER",
                "CASH_IN",
                "PAYMENT",
                "CASH_OUT",
                "TRANSFER",
                "CASH_IN",
                "PAYMENT",
            ],
            "nameOrig": [
                "C123",
                "C456",
                "C789",
                "C012",
                "C345",
                "C678",
                "C901",
                "C234",
                "C567",
                "C890",
                "C123",
                "C456",
            ],
            "nameDest": [
                "M345",
                "M678",
                "M901",
                "M234",
                "M567",
                "M890",
                "M123",
                "M456",
                "M789",
                "M012",
                "M345",
                "M678",
            ],
            "isFlaggedFraud": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "isFraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def model_server_ip():
    """Return the model server URL."""
    return MODEL_SERVER_IP


if __name__ == "__main__":
    model()
