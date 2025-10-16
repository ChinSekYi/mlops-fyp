"""
conftest.py

Shared pytest fixtures for FYP tests - simplified version
"""

import os

import joblib
import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from src.utils import load_environment

# Use ENV_FILE if set, otherwise default to .env
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
    """Load both model and preprocessor from MLflow (simplified from main.py)."""
    MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS")

    try:
        print(f"Loading model {MODEL_NAME} with alias {MODEL_ALIAS}")
        # Load model
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Successfully loaded MLflow model")

        # Load preprocessor from artifacts
        try:
            client = MlflowClient()
            model_version = client.get_model_version_by_alias(
                name=MODEL_NAME, alias=MODEL_ALIAS
            )
            run_id = model_version.run_id

            preprocessor_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="preprocessor"
            )

            preprocessor_files = os.listdir(preprocessor_path)
            preprocessor_file = None
            for file in preprocessor_files:
                if file.endswith(".pkl"):
                    preprocessor_file = os.path.join(preprocessor_path, file)
                    break

            if preprocessor_file:
                preprocessor = joblib.load(preprocessor_file)
                print("✅ Successfully loaded preprocessor from MLflow")
            else:
                preprocessor = None
                print("⚠️ No preprocessor found")

        except Exception as e:
            print(f"⚠️ Could not load preprocessor: {e}")
            preprocessor = None

        return model, preprocessor

    except Exception as e:
        print(f"❌ MLflow loading failed: {e}")

        # Simple dummy model for testing that can handle raw data
        class DummyModelWithPreprocessor:
            def predict(self, X):
                import numpy as np

                # Simple rule: fraud if amount > 200000
                return np.array(
                    [1 if row["amount"] > 200000 else 0 for _, row in X.iterrows()]
                )

        return DummyModelWithPreprocessor(), None


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
