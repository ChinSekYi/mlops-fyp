import os
import sys
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

    def initiate_model_trainer(self, train_array, test_array):
        try:
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

            return (training_data_accuracy, testing_data_accuracy)
        except Exception as e:
            raise CustomException(e, sys) from e
