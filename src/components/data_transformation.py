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
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=scaler,
            )
            logging.info("saved preprocessor")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            from src.exception import CustomException
            raise CustomException(e, None)
