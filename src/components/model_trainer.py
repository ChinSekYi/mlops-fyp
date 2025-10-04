import json
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, load_config, save_object

config = load_config()

trainer_config = config["model_trainer"]
trained_model_file_path = trainer_config["trained_model_file_path"]
metrics_file_path = trainer_config["metrics_file_path"]
processed_train_data_path = trainer_config["processed_train_data_path"]
processed_test_data_path = trainer_config["processed_test_data_path"]
registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
artifact_path = os.getenv("ARTIFACT_PATH")
"""
Model training module for the credit card fraud detection pipeline.
Trains, evaluates, and logs a logistic regression model.
"""


class ModelTrainer:
    """
    Handles model training, evaluation, and logging for the pipeline.
    """

    def __init__(self):
        self.trained_model_file_path = trained_model_file_path
        self.metrics_file_path = metrics_file_path
        os.makedirs(os.path.dirname(self.trained_model_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_file_path), exist_ok=True)

    def initiate_model_trainer(
        self,
        train_data_path,
        test_data_path,
        artifact_path_param,
        registered_model_name_param,
    ):
        """
        Trains a logistic regression model, evaluates it, logs to MLflow, and saves metrics.
        Args:
            train_data_path (str): Path to processed training data.
            test_data_path (str): Path to processed test data.
            artifact_path_param (str): MLflow artifact path.
            registered_model_name_param (str): MLflow registered model name.
        Returns:
            tuple: Training and testing metrics dictionaries.
        """
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Splitting training and test input data")
            x_train = train_df.drop(columns=["isFraud"])
            y_train = train_df["isFraud"]
            x_test = test_df.drop(columns=["isFraud"])
            y_test = test_df["isFraud"]

            """ 
            cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Naive Bayes', 3: 'Random Forest'}
            cv_models = [
                LogisticRegression(solver='liblinear', random_state=123, class_weight='balanced'),
                DecisionTreeClassifier(random_state=123),
                GaussianNB(),
                RandomForestClassifier(random_state=123)
            ]"""

            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("solver", "liblinear")

            # Train model
            # model = LogisticRegression(max_iter=1000)
            model = LogisticRegression(
                solver="liblinear", random_state=123, class_weight="balanced"
            )
            model.fit(x_train, y_train)

            # Save model locally and log to MLflow
            save_object(
                file_path=self.trained_model_file_path,
                obj=model,
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                input_example=x_train.iloc[[0]],
                name=artifact_path_param,
                registered_model_name=registered_model_name_param,
            )
            mlflow.log_artifact(self.trained_model_file_path)

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Directly compute metrics using sklearn
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

            # Log metrics to MLflow
            mlflow.log_metric(
                "train_accuracy", train_accuracy, model_id=model_info.model_id
            )
            mlflow.log_metric(
                "train_precision", train_precision, model_id=model_info.model_id
            )
            mlflow.log_metric(
                "train_recall", train_recall, model_id=model_info.model_id
            )
            mlflow.log_metric("train_f1", train_f1, model_id=model_info.model_id)
            mlflow.log_metric(
                "test_accuracy", test_accuracy, model_id=model_info.model_id
            )
            mlflow.log_metric(
                "test_precision", test_precision, model_id=model_info.model_id
            )
            mlflow.log_metric("test_recall", test_recall, model_id=model_info.model_id)
            mlflow.log_metric("test_f1", test_f1, model_id=model_info.model_id)

            # Save metrics.json for DVC tracking
            metrics_data = {
                "training_data_metrics": {
                    "accuracy": train_accuracy,
                    "precision": train_precision,
                    "recall": train_recall,
                    "f1": train_f1,
                },
                "testing_data_metrics": {
                    "accuracy": test_accuracy,
                    "precision": test_precision,
                    "recall": test_recall,
                    "f1": test_f1,
                },
            }
            print(metrics_data["testing_data_metrics"])
            return (
                metrics_data["training_data_metrics"],
                metrics_data["testing_data_metrics"],
            )

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    modeltrainer = ModelTrainer()
    training_metrics, testing_metrics = modeltrainer.initiate_model_trainer(
        processed_train_data_path,
        processed_test_data_path,
        artifact_path,
        registered_model_name,
    )

    print(f"training_data results: {training_metrics}")
    print(f"testing_data results: {testing_metrics}")
