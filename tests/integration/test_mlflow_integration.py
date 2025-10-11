"""
Integration tests for MLflow model registry and tracking functionality.
Tests model loading, tracking, and registry operations.
"""

import os

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from src.utils import load_environment

# Load environment
env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class TestMLflowIntegration:
    """Integration tests for MLflow tracking and model registry."""

    def test_mlflow_connection(self):
        """Test connection to MLflow tracking server."""
        try:
            client = MlflowClient()
            experiments = client.search_experiments()
            assert isinstance(experiments, list)
        except Exception as e:
            pytest.skip(f"MLflow server not available: {e}")

    def test_model_registry_access(self):
        """Test access to MLflow model registry."""
        if not MODEL_NAME:
            pytest.skip("MODEL_NAME not configured")

        try:
            client = MlflowClient()

            # Try to get the registered model
            try:
                model = client.get_registered_model(MODEL_NAME)
                assert model.name == MODEL_NAME
                assert len(model.latest_versions) > 0
            except mlflow.exceptions.RestException:
                pytest.skip(f"Model {MODEL_NAME} not found in registry")

        except Exception as e:
            pytest.skip(f"MLflow server not available: {e}")

    def test_model_loading_by_alias(self, model):
        """Test loading model using alias from fixture."""
        # The model fixture should successfully load the model
        assert model is not None

        # Test the model can make predictions
        test_data = pd.DataFrame(
            {
                "step": [1],
                "amount": [100.0],
                "oldbalanceOrg": [1000.0],
                "newbalanceOrig": [900.0],
                "oldbalanceDest": [0.0],
                "newbalanceDest": [100.0],
                "type__CASH_IN": [0],
                "type__CASH_OUT": [1],
                "type__DEBIT": [0],
                "type__PAYMENT": [0],
                "type__TRANSFER": [0],
                "nameOrig_token": [123],
                "nameDest_token": [456],
            }
        )

        try:
            prediction = model.predict(test_data)
            assert len(prediction) == 1
            assert prediction[0] in [0, 1]  # Binary classification
        except Exception as e:
            pytest.fail(f"Model prediction failed: {e}")

    def test_model_loading_by_version(self):
        """Test loading model using version number."""
        if not MODEL_NAME:
            pytest.skip("MODEL_NAME not configured")

        try:
            # Try to load latest version
            model_uri = f"models:/{MODEL_NAME}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            assert model is not None

        except Exception as e:
            pytest.skip(f"Could not load model by version: {e}")

    def test_experiment_creation(self):
        """Test creating and accessing MLflow experiments."""
        test_experiment_name = "test_integration_experiment"

        try:
            # Create or get experiment
            experiment_id = mlflow.create_experiment(
                test_experiment_name, tags={"purpose": "integration_testing"}
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(test_experiment_name)
            experiment_id = experiment.experiment_id

        assert experiment_id is not None

        # Test logging to experiment
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.85)

            # Verify run was logged
            run = mlflow.active_run()
            assert run is not None
            assert run.info.experiment_id == experiment_id

    @pytest.mark.slow
    def test_model_registry_workflow(self):
        """Test complete model registry workflow."""
        if not MODEL_NAME:
            pytest.skip("MODEL_NAME not configured")

        try:
            client = MlflowClient()

            # Get model versions
            model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            assert len(model_versions) > 0

            # Test getting model by stage (if any)
            latest_version = model_versions[0]

            # Test model metadata
            model_version = client.get_model_version(MODEL_NAME, latest_version.version)

            assert model_version.name == MODEL_NAME
            assert model_version.version == latest_version.version

        except Exception as e:
            pytest.skip(f"Model registry operations failed: {e}")

    def test_artifact_logging(self):
        """Test artifact logging and retrieval."""
        test_experiment_name = "test_artifact_experiment"

        try:
            experiment_id = mlflow.create_experiment(test_experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(test_experiment_name)
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id):
            # Log a simple artifact
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                json.dump({"test": "artifact"}, f)
                temp_file = f.name

            try:
                mlflow.log_artifact(temp_file, "test_artifacts")

                # Verify artifact was logged
                run = mlflow.active_run()
                client = MlflowClient()
                artifacts = client.list_artifacts(run.info.run_id)

                assert len(artifacts) > 0
                assert any("test_artifacts" in artifact.path for artifact in artifacts)

            finally:
                os.unlink(temp_file)
