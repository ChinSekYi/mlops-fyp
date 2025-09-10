import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
import mlflow
import mlflow.sklearn


class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")
    metrics_file_path = os.path.join("metrics", "metrics.json")
    mlflow_data_path = "mlflow_data"  # root folder for all MLflow data
    mlruns_path = os.path.join(mlflow_data_path, "mlruns")
    mlartifacts_path = os.path.join(mlflow_data_path, "mlartifacts")
    mlflow_db_path = os.path.join(mlflow_data_path, "mlflow.db")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

        os.makedirs(self.model_trainer_config.mlruns_path, exist_ok=True)
        os.makedirs(self.model_trainer_config.mlartifacts_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_trainer_config.metrics_file_path), exist_ok=True)

        # Set MLflow tracking URI to local MLflow server
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT"))

    def initiate_model_trainer(self, train_arr_path, test_arr_path):
        try:
            train_array = np.load(train_arr_path)
            test_array = np.load(test_arr_path)

            logging.info("Splitting training and test input data")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Start MLflow run
            with mlflow.start_run(run_name="logistic_regression_baseline"):
                
                # Log parameters
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("max_iter", 1000)

                # Train model
                model = LogisticRegression(max_iter=1000)
                model.fit(x_train, y_train)
                
                # Save model locally and log to MLflow
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=model,
                )

                # Create an example input from training features
                input_example = pd.DataFrame(x_train[:5], columns=[f"feature_{i}" for i in range(x_train.shape[1])])

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model", 
                    input_example=input_example
                )

                # Make predictions
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                # Evaluate
                training_metrics = evaluate_model(y_train, y_train_pred)
                testing_metrics = evaluate_model(y_test, y_test_pred)
                logging.info(f"testing_metrics: {testing_metrics}")

                # Log metrics to MLflow
                for key, value in training_metrics.items():
                    mlflow.log_metric(f"train_{key}", value)
                for key, value in testing_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)

                # Save metrics.json for DVC tracking
                os.makedirs("metrics", exist_ok=True)
                metrics_data = {
                    "training_data_metrics": training_metrics,
                    "testing_data_metrics": testing_metrics
                }
                with open(self.model_trainer_config.metrics_file_path, "w") as f:
                    json.dump(metrics_data, f)
            
            return training_metrics, testing_metrics
        
        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    modeltrainer = ModelTrainer()
    train_arr_path = "data/processed/train_transformed.npy"
    test_arr_path = "data/processed/test_transformed.npy"
    training_metrics, testing_metrics = modeltrainer.initiate_model_trainer(train_arr_path, test_arr_path)

    print(f"training_data results: {training_metrics}")
    print(f"testing_data results: {testing_metrics}")