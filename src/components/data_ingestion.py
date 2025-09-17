import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import mlflow

class DataIngestionConfig:
    train_data_path = os.path.join("data","processed", "train.csv")
    test_data_path = os.path.join("data", "processed", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, dataset_name):
        try:
            data_path = os.path.join("data", "raw", dataset_name)
            df = pd.read_csv(data_path)

            legit = df[df.Class == 0]
            fraud = df[df.Class == 1]
            
            legit_sample = legit.sample(n=492, random_state=2) # undersampling
            new_dataset = pd.concat([legit_sample, fraud], axis=0)
            
            X = new_dataset.drop(columns="Class", axis=1)
            Y = new_dataset["Class"]
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, stratify=Y, random_state=2
            )
            train = pd.concat([X_train, Y_train], axis=1)
            test = pd.concat([X_test, Y_test], axis=1)

            # mlflow dataset logging
            train_dataset = mlflow.data.from_pandas(df=train,source=dataset_name, name="train")
            mlflow.log_input(train_dataset, context="training") 
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train.to_csv(self.ingestion_config.train_data_path, index=False)
            test.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion completed.")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, None)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
