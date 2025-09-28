"""
Pipeline runner for the credit card fraud detection project.
Runs data ingestion, transformation, and model training with MLflow tracking.
"""

import os

import mlflow
from dotenv import load_dotenv

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

load_dotenv()

RUN_NAME = os.getenv("RUN_NAME")
DATASET_NAME = os.getenv("DATASET_NAME")
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
ALGORITHM_TYPE = os.getenv("ALGO_TYPE")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))

# Start MLflow run
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.set_tag("algorithm", ALGORITHM_TYPE)
    # ingestion
    print("Running DataIngestion")

    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # transformation
    print("Running DataTransformation")
    data_transformation = DataTransformation()
    train_arr_path, test_arr_path, _ = data_transformation.initiate_data_transformation(

        train_data_path, test_data_path
    )

    # training
    print("Running ModelTrainer")
    modeltrainer = ModelTrainer()

    # enter registered_model_name in model_trainer, if required
    training_metrics, testing_metrics = modeltrainer.initiate_model_trainer(
        train_arr_path, test_arr_path, ARTIFACT_PATH, REGISTERED_MODEL_NAME
    )

