import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import load_config

import mlflow
from mlflow.data.sources import LocalArtifactDatasetSource

config = load_config()

ingestion_config = config["data_ingestion"]
raw_data_path = ingestion_config["raw_data_path"]
train_data_path = ingestion_config["train_data_path"]
test_data_path = ingestion_config["test_data_path"]
test_size = ingestion_config["test_size"]
random_state = ingestion_config["random_state"]

class DataIngestion:
    def __init__(self):
        self.raw_data_path = raw_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.test_size = test_size
        self.random_state = random_state

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(self.raw_data_path)

            legit = df[df.Class == 0]
            fraud = df[df.Class == 1]
            
            legit_sample = legit.sample(n=492, random_state=self.random_state) # undersampling
            new_dataset = pd.concat([legit_sample, fraud], axis=0)
            
            X = new_dataset.drop(columns="Class", axis=1)
            Y = new_dataset["Class"]
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=self.test_size, stratify=Y, random_state=self.random_state
            )
            train = pd.concat([X_train, Y_train], axis=1)
            test = pd.concat([X_test, Y_test], axis=1)

            # mlflow dataset logging
            train_dataset = mlflow.data.from_pandas(df=train, source=LocalArtifactDatasetSource(self.raw_data_path), name="train")
            mlflow.log_input(train_dataset, context="training") 
            
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            train.to_csv(self.train_data_path, index=False)
            test.to_csv(self.test_data_path, index=False)
            logging.info("Data ingestion completed.")

            return self.train_data_path, self.test_data_path
        except Exception as e:
            raise CustomException(e, None)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion("creditcard.csv")
