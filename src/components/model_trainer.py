"""
Model training module for the credit card fraud detection pipeline.
Trains, evaluates, and logs a logistic regression model.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import load_config

config = load_config()

trainer_config = config["model_trainer"]
trained_model_file_path = trainer_config["trained_model_file_path"]
metrics_file_path = trainer_config["metrics_file_path"]
processed_train_data_path = trainer_config["processed_train_data_path"]
processed_test_data_path = trainer_config["processed_test_data_path"]
registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
artifact_path = os.getenv("ARTIFACT_PATH")


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
        registered_model_name_param,
        preprocessor_path=None,  # Add preprocessor path parameter
    ):
        """
        Train and evaluate multiple models, then log the best one to MLflow.

        Args:
            train_data_path: Path to training data CSV
            test_data_path: Path to test data CSV
            registered_model_name_param: Name for MLflow model registration
            preprocessor_path: Path to preprocessor pickle file

        Returns:
            dict: Training and testing metrics for all models
        """
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Splitting training and test input data")
            x_train = train_df.drop(columns=["isFraud"])
            y_train = train_df["isFraud"]
            x_test = test_df.drop(columns=["isFraud"])
            y_test = test_df["isFraud"]

            model_dict = {
                "LogisticRegression": LogisticRegression(
                    solver="liblinear", random_state=123, class_weight="balanced"
                ),
                "DecisionTree": DecisionTreeClassifier(random_state=123),
                "NaiveBayes": GaussianNB(),
                "RandomForest": RandomForestClassifier(random_state=123),
            }

            all_metrics = {}
            for model_name, model in model_dict.items():
                # with mlflow.start_run(run_name=model_name):
                with mlflow.start_run(
                    run_name=model_name, nested=mlflow.active_run() is not None
                ):
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_params(model.get_params())

                    # Cross-validation accuracy
                    cv_acc = cross_val_score(
                        model, x_train, y_train, cv=10, scoring="accuracy"
                    ).mean()
                    mlflow.log_metric("cv_accuracy", cv_acc)

                    # Train and predict
                    model.fit(x_train, y_train)
                    y_test_pred = model.predict(x_test)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    test_precision = precision_score(
                        y_test, y_test_pred, zero_division=0
                    )
                    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

                    mlflow.log_metric("test_accuracy", test_accuracy)
                    mlflow.log_metric("test_precision", test_precision)
                    mlflow.log_metric("test_recall", test_recall)
                    mlflow.log_metric("test_f1", test_f1)
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        input_example=x_train.iloc[[0]],
                        name=model_name,
                        registered_model_name=registered_model_name_param,
                    )

                    # Log the preprocessor as an artifact
                    if preprocessor_path and os.path.exists(preprocessor_path):
                        mlflow.log_artifact(preprocessor_path, "preprocessor")
                        logging.info(
                            f"Logged preprocessor artifact: " f"{preprocessor_path}"
                        )
                    else:
                        logging.warning(
                            "Preprocessor path not provided or file doesn't exist"
                        )

                    # Removed: mlflow.log_artifact(self.trained_model_file_path) - redundant with log_model

                    all_metrics[model_name] = {
                        "cv_accuracy": cv_acc,
                        "test_accuracy": test_accuracy,
                        "test_precision": test_precision,
                        "test_recall": test_recall,
                        "test_f1": test_f1,
                    }

            return all_metrics

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    modeltrainer = ModelTrainer()
    metrics = modeltrainer.initiate_model_trainer(
        processed_train_data_path,
        processed_test_data_path,
        registered_model_name,
    )

    print(f"training_data results: {metrics}")
