import mlflow
from mlflow import MlflowClient

import os
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from dotenv import load_dotenv
load_dotenv()

# Set MLflow tracking URI to local MLflow server
RUN_NAME = "logreg_v1"
DATASET_NAME = "creditcard.csv"
ARTIFACT_PATH = "logreg_model"
ALGORITHM_TYPE = "logistic-regression"
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")

# set MLFLOW_EXPERIMENT and MLFLOW_EXPERIMENT in .env
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT"))

# Start MLflow run
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.set_tag("algorithm", ALGORITHM_TYPE)
    
    # ingestion
    print(f'Running DataIngestion')
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion(DATASET_NAME)

    # transformation
    print(f'Running DataTransformation')
    data_transformation = DataTransformation()
    train_arr_path, test_arr_path, _  = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    # training
    print(f'Running ModelTrainer')
    modeltrainer = ModelTrainer()

    # enter registered_model_name in model_trainer, if required
    training_metrics, testing_metrics = modeltrainer.initiate_model_trainer(train_arr_path, test_arr_path, ARTIFACT_PATH, REGISTERED_MODEL_NAME)

    # set model registry tags
    # comment if this run is not saved in registry

""" 
client = MlflowClient()
client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    '21',
    "algorithm",
    ALGORITHM_TYPE
)
"""