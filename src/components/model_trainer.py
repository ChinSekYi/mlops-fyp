import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils import save_object, evaluate_model

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr_path, test_arr_path):
        try:
            train_array = np.load(train_arr_path)
            test_array = np.load(test_arr_path)

            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            model = LogisticRegression(max_iter=1000)
            model.fit(x_train, y_train)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )
            X_train_prediction = model.predict(x_train)
            X_test_prediction = model.predict(x_test)

            training_data_accuracy = evaluate_model(y_train, X_train_prediction)
            testing_data_accuracy = evaluate_model(y_test, X_test_prediction)
            logging.info(f'test_data_accuracy: {testing_data_accuracy}')

            metrics_dir = "metrics"
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "training_data_accuracy": training_data_accuracy,
                        "testing_data_accuracy": testing_data_accuracy
                    },
                    f
                )
            return (training_data_accuracy, testing_data_accuracy)
        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    modeltrainer = ModelTrainer()
    train_arr_path, test_arr_path = "data/processed/train_transformed.npy", "data/processed/test_transformed.npy",
    training_data_accuracy, testing_data_accuracy = modeltrainer.initiate_model_trainer(train_arr_path, test_arr_path)

    print(f"training_data results: {training_data_accuracy}")
    print(f"testing_data results: {testing_data_accuracy}")