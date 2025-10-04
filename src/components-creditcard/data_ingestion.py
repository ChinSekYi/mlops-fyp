"""
Data ingestion module for the credit card fraud detection pipeline.
Handles reading, sampling, splitting, and logging datasets.
"""

# flake8: noqa
import os

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import load_config, load_environment

# If running this script directly, uncomment the next line to ensure environment variables are loaded early
# load_environment(".env")
config = load_config()
DATASET_NAME = os.getenv("DATASET_NAME")

ingestion_config = config["data_ingestion"]
RAW_DATA_PATH_ = ingestion_config["raw_data_path"]
RAW_DATA_PATH = os.path.join(RAW_DATA_PATH_, DATASET_NAME)
TRAIN_DATA_PATH = ingestion_config["train_data_path"]
TEST_DATA_PATH = ingestion_config["test_data_path"]
TEST_SIZE = ingestion_config["test_size"]
RANDOM_STATE = ingestion_config["random_state"]


class DataIngestion:
    """
    Handles data ingestion: reading, sampling, splitting,
    and logging datasets for training/testing.
    """

    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH
        self.train_data_path = TRAIN_DATA_PATH
        self.test_data_path = TEST_DATA_PATH
        self.test_size = TEST_SIZE
        self.random_state = RANDOM_STATE

    def initiate_data_ingestion(self):
        """
        Reads the raw data, performs undersampling, splits into train/test,
        logs with MLflow, and saves files.
        Returns:
            tuple: Paths to train and test data CSV files.
        """
        try:
            df = pd.read_csv(self.raw_data_path)

            legit = df[df.Class == 0]
            fraud = df[df.Class == 1]

            legit_sample = legit.sample(
                n=492, random_state=self.random_state
            )  # undersampling
            new_dataset = pd.concat([legit_sample, fraud], axis=0)

            x = new_dataset.drop(columns="Class", axis=1)
            y = new_dataset["Class"]
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state,
            )
            train = pd.concat([x_train, y_train], axis=1)
            test = pd.concat([x_test, y_test], axis=1)

            if hasattr(mlflow.data, "from_pandas"):
                # Log train dataset
                train_dataset = mlflow.data.from_pandas(train, name="train")
                mlflow.log_input(train_dataset, context="training")

                # Log test dataset
                test_dataset = mlflow.data.from_pandas(test, name="test")
                mlflow.log_input(test_dataset, context="test")

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            train.to_csv(self.train_data_path, index=False)
            test.to_csv(self.test_data_path, index=False)
            logging.info("Data ingestion completed.")

            return self.train_data_path, self.test_data_path
        except Exception as e:
            raise CustomException(e, None) from e


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
