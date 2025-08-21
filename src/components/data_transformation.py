import os
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            scaler = StandardScaler()
            X_train = train_df.drop(columns=["Class"])
            y_train = train_df["Class"]
            X_test = test_df.drop(columns=["Class"])
            y_test = test_df["Class"]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            train_arr = np.c_[X_train_scaled, y_train.values]
            test_arr = np.c_[X_test_scaled, y_test.values]

            np.save("data/processed/train_transformed.npy", train_arr)
            np.save("data/processed/test_transformed.npy", test_arr)

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=scaler,
            )

            logging.info("saved preprocessor")

            return (
                "data/processed/train_transformed.npy", 
                "data/processed/test_transformed.npy",
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            from src.exception import CustomException
            raise CustomException(e, None)

if __name__ == "__main__":
    train_data_path, test_data_path = "data/processed/train.csv", "data/processed/test.csv",
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )
