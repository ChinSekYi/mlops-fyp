import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, load_config
import mlflow
import mlflow.sklearn

load_dotenv()

config = load_config()

trainer_config = config["model_trainer"]
trained_model_file_path = trainer_config["trained_model_file_path"]
metrics_file_path = trainer_config["metrics_file_path"]
processed_train_data_path = trainer_config["processed_train_data_path"]
processed_test_data_path = trainer_config["processed_test_data_path"]
artifact_path = trainer_config.get("artifact_path", "LogisticRegressionModelv1")
registered_model_name = trainer_config.get("registered_model_name", None)

class ModelTrainer:
    def __init__(self):
        self.trained_model_file_path = trained_model_file_path
        self.metrics_file_path = metrics_file_path

        os.makedirs(os.path.dirname(self.trained_model_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_file_path), exist_ok=True)

    def initiate_model_trainer(self, processed_train_data_path, processed_test_data_path, artifact_path, registered_model_name):
        try:
            train_df = pd.read_csv(processed_train_data_path)
            test_df = pd.read_csv(processed_test_data_path)

            logging.info("Splitting training and test input data")
            X_train = train_df.drop(columns=["Class"])
            y_train = train_df["Class"]
            X_test = test_df.drop(columns=["Class"])
            y_test = test_df["Class"]

            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)

            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Save model locally and log to MLflow
            save_object(
                file_path=self.trained_model_file_path,
                obj=model,
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                input_example=X_train.iloc[[0]],
                name=artifact_path,
                registered_model_name=registered_model_name
            )
            mlflow.log_artifact(self.trained_model_file_path)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate
            training_metrics = evaluate_model(y_train, y_train_pred)
            testing_metrics = evaluate_model(y_test, y_test_pred)
            logging.info(f"testing_metrics: {testing_metrics}")

            # Log metrics to MLflow
            for key, value in training_metrics.items():
                mlflow.log_metric(f"train_{key}", value, model_id=model_info.model_id)
            for key, value in testing_metrics.items():
                mlflow.log_metric(f"test_{key}", value, model_id=model_info.model_id)
            
            # Save metrics.json for DVC tracking
            os.makedirs(os.path.dirname(self.metrics_file_path), exist_ok=True)
            metrics_data = {
                "training_data_metrics": training_metrics,
                "testing_data_metrics": testing_metrics
            }
            with open(self.metrics_file_path, "w") as f:
                json.dump(metrics_data, f)
            
            return training_metrics, testing_metrics
        
        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    modeltrainer = ModelTrainer()
    training_metrics, testing_metrics = modeltrainer.initiate_model_trainer(
        processed_train_data_path, processed_test_data_path, artifact_path, registered_model_name
    )

    print(f"training_data results: {training_metrics}")
    print(f"testing_data results: {testing_metrics}")