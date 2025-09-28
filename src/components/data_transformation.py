"""
Data transformation module for the credit card fraud detection pipeline.
Scales features and saves preprocessor and processed datasets.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import load_config, save_object

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
            tuple: Paths to processed train, test data, and preprocessor object file.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            scaler = StandardScaler()
            x_train = train_df.drop(columns=["Class"])
            y_train = train_df["Class"]
            x_test = test_df.drop(columns=["Class"])
            y_test = test_df["Class"]

            feature_columns = x_train.columns.tolist()
            all_columns = feature_columns + ["Class"]

            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            train_arr = np.c_[x_train_scaled, y_train.values]
            test_arr = np.c_[x_test_scaled, y_test.values]

            processed_train_df = pd.DataFrame(train_arr, columns=all_columns)
            processed_test_df = pd.DataFrame(test_arr, columns=all_columns)
            processed_train_df.to_csv(self.processed_train_data_path, index=False)
            processed_test_df.to_csv(self.processed_test_data_path, index=False)

            save_object(
                file_path=self.preprocessor_ob_file_path,
                obj=scaler,
            )

            mlflow.log_artifact(self.preprocessor_ob_file_path)
            logging.info("saved preprocessor and processed datasets")
            return (
                self.processed_train_data_path,
                self.processed_test_data_path,
                self.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e, None) from e


if __name__ == "__main__":
    data_transformation = DataTransformation()
    processed_train_data_path, processed_test_data_path, _ = (
        data_transformation.initiate_data_transformation(
            processed_train_data_path, processed_test_data_path
        )
    )
