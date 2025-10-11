"""
Integration tests for the ML pipeline components.
Tests the interaction between different pipeline stages with real data flow.
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TestPipelineIntegration:
    """Test the integration between pipeline components with realistic data flow."""

    def test_data_ingestion_to_transformation_integration(self, mock_raw_df):
        """Test that data ingestion output can be processed by data transformation."""

        test_data = mock_raw_df.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data
            raw_data_path = os.path.join(temp_dir, "raw_data.csv")
            test_data.to_csv(raw_data_path, index=False)

            # Mock data ingestion to use our test data
            with patch.object(DataIngestion, "__init__", lambda self: None):
                ingestion = DataIngestion()
                ingestion.raw_data_path = raw_data_path
                ingestion.train_data_path = os.path.join(temp_dir, "train.csv")
                ingestion.test_data_path = os.path.join(temp_dir, "test.csv")
                ingestion.test_size = 0.2
                ingestion.random_state = 42

                # Run data ingestion
                train_path, test_path = ingestion.initiate_data_ingestion()

                # Verify files were created
                assert os.path.exists(train_path)
                assert os.path.exists(test_path)

                # Verify data can be read
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                assert len(train_df) > 0
                assert len(test_df) > 0
                assert "isFraud" in train_df.columns
                assert "isFraud" in test_df.columns

                # Now test transformation can process this data
                with patch.object(DataTransformation, "__init__", lambda self: None):
                    transformation = DataTransformation()
                    transformation.processed_train_data_path = os.path.join(
                        temp_dir, "processed_train.csv"
                    )
                    transformation.processed_test_data_path = os.path.join(
                        temp_dir, "processed_test.csv"
                    )
                    transformation.preprocessor_ob_file_path = os.path.join(
                        temp_dir, "preprocessor.pkl"
                    )

                    # Run transformation
                    processed_train_path, processed_test_path, _ = (
                        transformation.initiate_data_transformation(
                            train_path, test_path
                        )
                    )

                    # Verify processed files exist
                    assert os.path.exists(processed_train_path)
                    assert os.path.exists(processed_test_path)

                    # Verify processed data structure
                    processed_train_df = pd.read_csv(processed_train_path)

                    # Should have one-hot encoded type columns
                    type_columns = [
                        col
                        for col in processed_train_df.columns
                        if col.startswith("type__")
                    ]
                    assert len(type_columns) > 0

                    # Should have tokenized name columns
                    assert "nameOrig_token" in processed_train_df.columns
                    assert "nameDest_token" in processed_train_df.columns

                    # Should not have original categorical columns
                    assert "type" not in processed_train_df.columns
                    assert "nameOrig" not in processed_train_df.columns
                    assert "nameDest" not in processed_train_df.columns
                    assert "isFlaggedFraud" not in processed_train_df.columns

    @pytest.mark.slow
    def test_full_pipeline_integration(self):
        """Test the complete pipeline from ingestion to model training."""

        # Create larger, more balanced test dataset for model training
        n_samples = 200
        # Create more balanced fraud distribution
        fraud_indices = list(range(0, n_samples, 4)) + list(
            range(1, n_samples, 7)
        )  # More varied pattern
        fraud_labels = [1 if i in fraud_indices else 0 for i in range(n_samples)]

        test_data = pd.DataFrame(
            {
                "step": list(range(1, n_samples + 1)),
                "amount": [100.0 + i * 10 + (i % 3) * 50 for i in range(n_samples)],
                "oldbalanceOrg": [
                    1000.0 + i * 100 + (i % 5) * 200 for i in range(n_samples)
                ],
                "newbalanceOrig": [
                    900.0 + i * 90 + (i % 4) * 150 for i in range(n_samples)
                ],
                "oldbalanceDest": [i * 50 + (i % 6) * 75 for i in range(n_samples)],
                "newbalanceDest": [
                    100.0 + i * 60 + (i % 7) * 80 for i in range(n_samples)
                ],
                "type": ["CASH_OUT", "TRANSFER", "CASH_IN", "PAYMENT", "DEBIT"]
                * (n_samples // 5),
                "nameOrig": [f"C{i:03d}" for i in range(n_samples)],
                "nameDest": [f"M{i:03d}" for i in range(n_samples)],
                "isFlaggedFraud": [0] * n_samples,
                "isFraud": fraud_labels,
            }
        )

        # Verify we have both classes
        fraud_counts = pd.Series(fraud_labels).value_counts()
        assert len(fraud_counts) == 2, f"Need both fraud classes, got: {fraud_counts}"
        assert (
            fraud_counts.min() >= 10
        ), f"Need at least 10 samples per class, got: {fraud_counts}"

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_data_path = os.path.join(temp_dir, "raw_data.csv")
            test_data.to_csv(raw_data_path, index=False)

            # Step 1: Data Ingestion
            with patch.object(DataIngestion, "__init__", lambda self: None):
                ingestion = DataIngestion()
                ingestion.raw_data_path = raw_data_path
                ingestion.train_data_path = os.path.join(temp_dir, "train.csv")
                ingestion.test_data_path = os.path.join(temp_dir, "test.csv")
                ingestion.test_size = 0.2
                ingestion.random_state = 42

                train_path, test_path = ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            with patch.object(DataTransformation, "__init__", lambda self: None):
                transformation = DataTransformation()
                transformation.processed_train_data_path = os.path.join(
                    temp_dir, "processed_train.csv"
                )
                transformation.processed_test_data_path = os.path.join(
                    temp_dir, "processed_test.csv"
                )
                transformation.preprocessor_ob_file_path = os.path.join(
                    temp_dir, "preprocessor.pkl"
                )
                processed_train_path, processed_test_path, _ = (
                    transformation.initiate_data_transformation(train_path, test_path)
                )

            # Step 3: Model Training
            with patch.object(ModelTrainer, "__init__", lambda self: None), patch(
                "src.components.model_trainer.mlflow.start_run"
            ), patch("src.components.model_trainer.mlflow.log_param"), patch(
                "src.components.model_trainer.mlflow.log_params"
            ), patch(
                "src.components.model_trainer.mlflow.log_metric"
            ), patch(
                "src.components.model_trainer.mlflow.sklearn.log_model"
            ), patch(
                "src.components.model_trainer.cross_val_score",
                return_value=np.array(
                    [0.8, 0.85, 0.82, 0.88, 0.83, 0.87, 0.84, 0.86, 0.81, 0.89]
                ),
            ):

                trainer = ModelTrainer()
                trainer.trained_model_file_path = os.path.join(temp_dir, "model.pkl")
                trainer.metrics_file_path = os.path.join(temp_dir, "metrics.json")

                # This should work without errors
                metrics = trainer.initiate_model_trainer(
                    processed_train_path, processed_test_path, "test_model"
                )

                # Verify metrics were returned
                assert isinstance(metrics, dict)
                assert len(metrics) > 0

                # Verify metrics have expected structure
                for model_name, model_metrics in metrics.items():
                    assert "cv_accuracy" in model_metrics
                    assert "test_accuracy" in model_metrics
                    assert "test_precision" in model_metrics
                    assert "test_recall" in model_metrics
                    assert "test_f1" in model_metrics
