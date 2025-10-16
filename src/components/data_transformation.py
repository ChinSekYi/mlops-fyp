"""
Data transformation module for the credit card fraud detection pipeline.
Scales features and saves preprocessor and processed datasets.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import balance_classes_smotenc, load_config, save_object, tokenize_column

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

    def get_data_transformer_object(self):
        """
        Creates a preprocessing pipeline for PaySim transaction data.
        Note: This assumes tokenization is done BEFORE this preprocessor.
        For a complete pipeline, tokenization should be included here.
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        try:
            # Define feature groups (AFTER tokenization has been applied)
            numerical_features = [
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "nameOrig_token",  # These are created by tokenization
                "nameDest_token",  # These are created by tokenization
            ]

            categorical_features = ["type"]

            # Create preprocessing pipelines
            num_pipeline = Pipeline([("scaler", StandardScaler())])

            cat_pipeline = Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(
                            categories=[
                                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
                            ],
                            drop=None,
                            sparse_output=False,
                        ),
                    )
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_features),
                    ("cat", cat_pipeline, categorical_features),
                ]
            )

            logging.info("Preprocessing pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, None) from e

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test data, creates and fits preprocessor, transforms data, and saves everything.
        Args:
            train_path (str): Path to the training data CSV.
            test_path (str): Path to the test data CSV.
        Returns:
            tuple: Paths to processed train and test data, and preprocessor object path.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Separate features and target
            x_train = train_df.drop(columns=["isFraud"])
            y_train = train_df["isFraud"]
            x_test = test_df.drop(columns=["isFraud"])
            y_test = test_df["isFraud"]

            # Drop unnecessary columns
            x_train = x_train.drop(["isFlaggedFraud"], axis=1)
            x_test = x_test.drop(["isFlaggedFraud"], axis=1)

            # Manual preprocessing for tokenization (before using sklearn pipeline)
            # Tokenize name columns
            x_train, x_test = tokenize_column(x_train, x_test, "nameOrig")
            x_train, x_test = tokenize_column(x_train, x_test, "nameDest")

            # Drop original name columns and keep tokenized versions
            x_train = x_train.drop(["nameOrig", "nameDest"], axis=1)
            x_test = x_test.drop(["nameOrig", "nameDest"], axis=1)

            # Create and fit the preprocessor
            preprocessor = self.get_data_transformer_object()

            # Fit preprocessor on training data
            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed = preprocessor.transform(x_test)

            # Convert back to DataFrame for easier handling
            feature_names = [
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "nameOrig_token",
                "nameDest_token",
            ] + [
                f"type__{cat}"
                for cat in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
            ]

            x_train_df = pd.DataFrame(x_train_transformed, columns=feature_names)
            x_test_df = pd.DataFrame(x_test_transformed, columns=feature_names)

            # Handle class imbalance with SMOTENC
            logging.info(f"Fraud rates before balancing: {y_train.mean():.4f}")
            categorical_features = []
            type_columns = [
                col for col in x_train_df.columns if col.startswith("type__")
            ]
            for col in type_columns:
                categorical_features.append(x_train_df.columns.get_loc(col))

            x_train_balanced, y_train_balanced = balance_classes_smotenc(
                x_train_df, y_train, categorical_features
            )
            logging.info(f"Fraud rates after balancing: {y_train_balanced.mean():.4f}")

            # Combine features and target for saving
            train_final = pd.concat(
                [
                    x_train_balanced.reset_index(drop=True),
                    y_train_balanced.reset_index(drop=True),
                ],
                axis=1,
            )
            test_final = pd.concat(
                [x_test_df.reset_index(drop=True), y_test.reset_index(drop=True)],
                axis=1,
            )

            # Save processed data
            train_final.to_csv(self.processed_train_data_path, index=False)
            test_final.to_csv(self.processed_test_data_path, index=False)

            # Save the preprocessor object
            save_object(file_path=self.preprocessor_ob_file_path, obj=preprocessor)

            logging.info(
                "Data transformation completed. Processed datasets and preprocessor saved."
            )
            return (
                self.processed_train_data_path,
                self.processed_test_data_path,
                self.preprocessor_ob_file_path,
                preprocessor,  # Return the fitted preprocessor object
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
    (
        processed_train_data_path,
        processed_test_data_path,
        preprocessor_path,
        preprocessor_obj,
    ) = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Preprocessor object type: {type(preprocessor_obj)}")
