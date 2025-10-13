"""
Data transformation module for the credit card fraud detection pipeline.
Scales features and saves preprocessor and processed datasets.
"""

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    balance_classes_smotenc,
    load_config,
    one_hot_encode_and_align,
    standardize_columns,
    tokenize_column,
)

config = load_config()
transformation_config = config["data_transformation"]
processed_train_data_path = transformation_config["processed_train_data_path"]
processed_test_data_path = transformation_config["processed_test_data_path"]
preprocessor_ob_file_path = transformation_config["preprocessor_ob_file_path"]


class DataTransformation:
    """
    Handles feature scaling and saving of preprocessor and processed datasets.
    """

    def __init__(self):
        self.processed_train_data_path = processed_train_data_path
        self.processed_test_data_path = processed_test_data_path
        self.preprocessor_ob_file_path = preprocessor_ob_file_path

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test data, scales features, saves processed data and scaler, logs artifact.
        Args:
            train_path (str): Path to the training data CSV.
            test_path (str): Path to the test data CSV.
        Returns:
            tuple: Paths to processed train, and test data.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            x_train = train_df.drop(columns=["isFraud"])
            y_train = train_df["isFraud"]
            x_test = test_df.drop(columns=["isFraud"])
            y_test = test_df["isFraud"]

            # Drop col
            x_train = x_train.drop(["isFlaggedFraud"], axis=1)
            x_test = x_test.drop(["isFlaggedFraud"], axis=1)

            # One hot encode 'Type' column
            x_train, x_test = one_hot_encode_and_align(x_train, x_test, "type")

            # Standardisation (Normalisation)
            col_names = [
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
            ]
            x_train, x_test = standardize_columns(x_train, x_test, col_names)

            # Handle imbalance
            # print(f"Fraud rates before balancing: {y_train.mean():.4f}")
            categorical_features = []
            type_columns = [col for col in x_train.columns if col.startswith("type__")]
            for col in type_columns:
                categorical_features.append(x_train.columns.get_loc(col))

            # Add indices for original categorical columns (nameOrig, nameDest)
            categorical_features.append(x_train.columns.get_loc("nameOrig"))
            categorical_features.append(x_train.columns.get_loc("nameDest"))

            x_train, y_train = balance_classes_smotenc(
                x_train, y_train, categorical_features
            )

            # Tokenisation to convert categorical text data to numerical features. Eg "C1231006815" -> 12345
            x_train, x_test = tokenize_column(x_train, x_test, "nameOrig")
            x_train, x_test = tokenize_column(x_train, x_test, "nameDest")

            # Drop col
            x_train = x_train.drop(["nameOrig", "nameDest"], axis=1)
            x_test = x_test.drop(["nameOrig", "nameDest"], axis=1)

            x_train = x_train.reset_index(drop=True)
            x_train = pd.concat([x_train, y_train], axis=1)

            x_test = x_test.reset_index(drop=True)
            x_test = pd.concat([x_test, y_test], axis=1)

            x_train.to_csv(self.processed_train_data_path, index=False)
            x_test.to_csv(self.processed_test_data_path, index=False)

            logging.info(
                "data transformation completed.  processed_test.csv and processed_train.csv saved."
            )
            return (
                self.processed_train_data_path,
                self.processed_test_data_path,
                self.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e, None) from e


if __name__ == "__main__":
    # Use the correct input file paths for train and test data
    config = load_config()
    ingestion_config = config["data_ingestion"]
    train_data_path = ingestion_config["train_data_path"]
    test_data_path = ingestion_config["test_data_path"]
    print(train_data_path, test_data_path)

    data_transformation = DataTransformation()
    processed_train_data_path, processed_test_data_path, _ = (
        data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
    )
