"""
Pipeline runner for the credit card fraud detection project.
Runs data ingestion, transformation, and model training with MLflow tracking.
"""

# flake8: noqa
import os

import mlflow

from src.utils import load_environment

# Use ENV_FILE if set, otherwise default to .env.dev
env_file = os.getenv("ENV_FILE", ".env.ci")
load_environment(env_file)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

RUN_NAME = os.getenv("RUN_NAME")
DATASET_NAME = os.getenv("DATASET_NAME")
#ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
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
    all_metrics = modeltrainer.initiate_model_trainer(
        train_arr_path, test_arr_path, REGISTERED_MODEL_NAME
    )
