import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv("data/creditcard.csv")
            legit = df[df.Class == 0]
            fraud = df[df.Class == 1]
            legit_sample = legit.sample(n=492, random_state=2)
            new_dataset = pd.concat([legit_sample, fraud], axis=0)
            X = new_dataset.drop(columns="Class", axis=1)
            Y = new_dataset["Class"]
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, stratify=Y, random_state=2
            )
            train = pd.concat([X_train, Y_train], axis=1)
            test = pd.concat([X_test, Y_test], axis=1)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train.to_csv(self.ingestion_config.train_data_path, index=False)
            test.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion completed.")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, None)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    training_data_accuracy, testing_data_accuracy = modeltrainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Training data size: {len(training_data_accuracy)}")
    print(f"training_data results: {training_data_accuracy}")
    print(f"testing_data results: {testing_data_accuracy}")
