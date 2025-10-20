"""
Unit tests for the ML pipeline
Tests data ingestion, transformation, and model training components.

Note:
# return_value - same result every call
mock_function.return_value = "hello"
mock_function()  # Returns "hello"
mock_function()  # Returns "hello" again

# side_effect - different results per call
mock_function.side_effect = ["hello", "world"]
mock_function()  # Returns "hello"
mock_function()  # Returns "world"
mock_function()  # Raises StopIteration (no more values)

MagicMock:
MagicMock is a powerful testing tool that creates "fake" objects which can pretend to be anything.
It automatically handles any method call, attribute access, or operation you perform on it.
Unlike regular Mock, MagicMock supports Python's "magic methods" like __enter__, __exit__ (for context managers),
__len__, __getitem__, etc. This makes it perfect for mocking complex objects like MLflow runs, database
connections, or any object that uses special Python protocols. In our tests, we use MagicMock to:
- Mock MLflow context managers (with statements)
- Create fake objects that respond to any method call
- Prevent actual external service connections during testing
- Ensure tests run fast and reliably without side effects
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Mock DataFrame that represents processed data (after transformation) for model training tests
mock_processed_df = pd.DataFrame(
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
        # One-hot encoded type columns (after transformation)
        "type_CASH_IN": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        "type_CASH_OUT": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "type_PAYMENT": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        "type_TRANSFER": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        # Tokenized columns (final result)
        "nameOrig_token": [123, 456, 789, 12, 345, 678, 901, 234, 567, 890, 123, 456],
        "nameDest_token": [345, 678, 901, 234, 567, 890, 123, 456, 789, 12, 345, 678],
        "isFraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
)


class TestDataIngestion:
    """Test the DataIngestion component."""

    def test_data_ingestion_init(self):
        """Test DataIngestion initialization."""
        ingestion = DataIngestion()
        assert ingestion is not None
        assert hasattr(ingestion, "raw_data_path")
        assert hasattr(ingestion, "train_data_path")
        assert hasattr(ingestion, "test_data_path")
        assert hasattr(ingestion, "test_size")
        assert hasattr(ingestion, "random_state")

    @patch("src.components.data_ingestion.pd.read_csv")
    @patch("src.components.data_ingestion.train_test_split")
    def test_initiate_data_ingestion(self, mock_split, mock_read_csv, mock_raw_df):
        """Test data ingestion process with mocked dependencies."""
        # Mock pd.read_csv to return an iterator of DataFrames (chunks)
        mock_read_csv.return_value = iter([mock_raw_df])

        # Mock train_test_split
        mock_split.return_value = (
            mock_raw_df.iloc[:2],
            mock_raw_df.iloc[2:],  # X_train, X_test
            pd.Series([0, 1]),
            pd.Series([0, 1]),  # y_train, y_test
        )

        # Test
        ingestion = DataIngestion()
        with patch.object(ingestion, "train_data_path", "/tmp/train.csv"), patch.object(
            ingestion, "test_data_path", "/tmp/test.csv"
        ), patch("pandas.DataFrame.to_csv"), patch("os.makedirs"):

            train_path, test_path = ingestion.initiate_data_ingestion()

            assert train_path == ingestion.train_data_path
            assert test_path == ingestion.test_data_path
            # These assertions ensure your code is actually using the functions it's supposed to use
            mock_read_csv.assert_called_once()  # Verifies that pd.read_csv() was called exactly one time during the test
            mock_split.assert_called_once()
            # Note: balance_classes is now called in data_transformation, not data_ingestion


class TestDataTransformation:
    """Test the DataTransformation component."""

    def test_data_transformation_init(self):
        """Test DataTransformation initialization."""
        transformation = DataTransformation()
        assert transformation is not None
        assert hasattr(transformation, "processed_train_data_path")
        assert hasattr(transformation, "processed_test_data_path")

    @patch("src.components.data_transformation.pd.read_csv")
    @patch("src.components.data_transformation.tokenize_column")
    @patch("src.components.data_transformation.balance_classes_smotenc")
    def test_initiate_data_transformation(
        self, mock_balance, mock_tokenize, mock_read_csv, mock_raw_df
    ):
        # Mock CSV reading
        mock_read_csv.side_effect = [mock_raw_df, mock_raw_df]

        # Create DataFrame with tokenized columns (simulating tokenize_column output)
        mock_after_tokenize_df = mock_raw_df.copy()
        # Generate token values for 50 rows (matching the updated fixture size)
        mock_after_tokenize_df["nameOrig_token"] = [
            i + 100 for i in range(len(mock_raw_df))
        ]
        mock_after_tokenize_df["nameDest_token"] = [
            i + 200 for i in range(len(mock_raw_df))
        ]

        # Mock tokenize_column to return tokenized data
        mock_tokenize.return_value = (mock_after_tokenize_df, mock_after_tokenize_df)

        # Mock balance_classes_smotenc
        mock_balance.return_value = (
            mock_after_tokenize_df.iloc[:6],
            pd.Series([0, 1, 0, 1, 0, 1]),
        )

        # Test
        transformation = DataTransformation()
        with patch.object(
            transformation, "processed_train_data_path", "/tmp/processed_train.csv"
        ), patch.object(
            transformation, "processed_test_data_path", "/tmp/processed_test.csv"
        ), patch.object(
            transformation, "preprocessor_ob_file_path", "/tmp/preprocessor.pkl"
        ), patch(
            "pandas.DataFrame.to_csv"
        ), patch(
            "os.makedirs"
        ), patch(
            "src.components.data_transformation.save_object"
        ):

            train_path, test_path, preprocessor_path, _ = (
                transformation.initiate_data_transformation(
                    "/tmp/train.csv", "/tmp/test.csv"
                )
            )

            assert train_path == transformation.processed_train_data_path
            assert test_path == transformation.processed_test_data_path
            assert preprocessor_path == transformation.preprocessor_ob_file_path
            assert mock_read_csv.call_count == 2
            # tokenize_column should be called twice (once for nameOrig, once for nameDest)
            assert mock_tokenize.call_count == 2
            mock_balance.assert_called_once()


class TestModelTrainer:
    """Test the ModelTrainer component."""

    def test_model_trainer_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert trainer is not None
        assert hasattr(trainer, "trained_model_file_path")
        assert hasattr(trainer, "metrics_file_path")

    @patch("src.components.model_trainer.pd.read_csv")
    @patch("src.components.model_trainer.mlflow.start_run")
    @patch("src.components.model_trainer.cross_val_score")
    def test_initiate_model_trainer(
        self, mock_cross_val_score, mock_start_run, mock_read_csv
    ):
        """Test model training process with simplified mocking."""
        # side_effect allows a mock to return different values on successive calls
        mock_read_csv.side_effect = [mock_processed_df, mock_processed_df]

        # Mock cross_val_score to return numpy array (not list) so .mean() works
        import numpy as np

        mock_cross_val_score.return_value = np.array(
            [0.8, 0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.89, 0.86, 0.90]
        )

        # Mock MLflow to do nothing
        mock_start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Test
        trainer = ModelTrainer()
        with patch("os.makedirs"), patch(
            "src.components.model_trainer.mlflow.log_param"
        ), patch("src.components.model_trainer.mlflow.log_params"), patch(
            "src.components.model_trainer.mlflow.log_metric"
        ), patch(
            "src.components.model_trainer.mlflow.sklearn.log_model"
        ), patch(
            "sklearn.linear_model.LogisticRegression.fit"
        ), patch(
            "sklearn.tree.DecisionTreeClassifier.fit"
        ), patch(
            "sklearn.naive_bayes.GaussianNB.fit"
        ), patch(
            "sklearn.ensemble.RandomForestClassifier.fit"
        ), patch(
            "sklearn.linear_model.LogisticRegression.predict",
            return_value=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ), patch(
            "sklearn.tree.DecisionTreeClassifier.predict",
            return_value=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ), patch(
            "sklearn.naive_bayes.GaussianNB.predict",
            return_value=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ), patch(
            "sklearn.ensemble.RandomForestClassifier.predict",
            return_value=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ):

            metrics = trainer.initiate_model_trainer(
                "/tmp/train.csv", "/tmp/test.csv", "test_model"
            )

            # Basic assertions
            assert isinstance(metrics, dict)
            assert mock_read_csv.call_count == 2
            # Verify cross_val_score was called for each model (4 models)
            assert mock_cross_val_score.call_count >= 4


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_pipeline_components_exist(self):
        """Test that all pipeline components can be imported and instantiated."""
        # Test component imports
        ingestion = DataIngestion()
        transformation = DataTransformation()
        trainer = ModelTrainer()

        assert ingestion is not None
        assert transformation is not None
        assert trainer is not None

        # Test they have required methods
        assert hasattr(ingestion, "initiate_data_ingestion")
        assert hasattr(transformation, "initiate_data_transformation")
        assert hasattr(trainer, "initiate_model_trainer")
        assert callable(ingestion.initiate_data_ingestion)
        assert callable(transformation.initiate_data_transformation)
        assert callable(trainer.initiate_model_trainer)
