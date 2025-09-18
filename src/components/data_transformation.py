import os
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from src.utils import save_object
import mlflow

class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.processed_train_data_path = os.path.join("data","processed", "processed_train.csv")
        self.processed_test_data_path = os.path.join("data", "processed", "processed_test.csv")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            scaler = StandardScaler()
            X_train = train_df.drop(columns=["Class"])
            y_train = train_df["Class"]
            X_test = test_df.drop(columns=["Class"])
            y_test = test_df["Class"]

            feature_columns = X_train.columns.tolist()  # or use the list directly if you know the order
            all_columns = feature_columns + ["Class"]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            train_arr = np.c_[X_train_scaled, y_train.values]
            test_arr = np.c_[X_test_scaled, y_test.values]

            processed_train_df = pd.DataFrame(train_arr, columns=all_columns)
            processed_test_df = pd.DataFrame(test_arr, columns=all_columns)
            processed_train_df.to_csv(self.processed_train_data_path, index=False)
            processed_test_df.to_csv(self.processed_test_data_path, index=False)

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=scaler,
            ) # TODO: might not be necessary because it is saved to mlflow server. but predict pipeline might need it. 
            
            mlflow.log_artifact("artifacts/preprocessor.pkl")
            logging.info("saved preprocessor and processed datasets")
            return (
                self.processed_train_data_path, 
                self.processed_test_data_path, 
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            from src.exception import CustomException
            raise CustomException(e, None)

if __name__ == "__main__":
    train_data_path, test_data_path = "data/processed/train.csv", "data/processed/test.csv"

    data_transformation = DataTransformation()
    processed_train_data_path, processed_test_data_path, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )
