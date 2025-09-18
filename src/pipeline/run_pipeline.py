import mlflow
from mlflow import MlflowClient

import os
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from dotenv import load_dotenv
load_dotenv()

# Set MLflow tracking URI to local MLflow server
RUN_NAME = "xgboost_v8"
DATASET_NAME = "creditcard.csv"
ARTIFACT_PATH = "xgboost_model"
ALGORITHM_TYPE = "xgboost"
REGISTERED_MODEL_NAME = "fraud-detection-model"

# set MLFLOW_EXPERIMENT and MLFLOW_EXPERIMENT in .env
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT"))

# Start MLflow run
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.set_tag("algorithm", ALGORITHM_TYPE)
    
    # ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion(DATASET_NAME)

    # transformation
    data_transformation = DataTransformation()
    train_arr_path, test_arr_path, _  = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    # training
    modeltrainer = ModelTrainer()

    # enter registered_model_name in model_trainer, if required
    training_metrics, testing_metrics = modeltrainer.initiate_model_trainer(train_arr_path, test_arr_path, ARTIFACT_PATH, REGISTERED_MODEL_NAME)

    # set model registry tags
    # comment if this run is not saved in registry

client = MlflowClient()
client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    '21',
    "algorithm",
    ALGORITHM_TYPE
)