import mlflow
import os
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.utils import load_config
from dotenv import load_dotenv
load_dotenv()

config = load_config()
pipeline_config = config["run_pipeline"]

RUN_NAME = pipeline_config["run_name"]
DATASET_NAME = pipeline_config["dataset_name"]
ARTIFACT_PATH = pipeline_config["artifact_path"]
ALGORITHM_TYPE = pipeline_config["algorithm_type"]
REGISTERED_MODEL_NAME = pipeline_config["registered_model_name"]

# set MLFLOW_EXPERIMENT and MLFLOW_EXPERIMENT in .env
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(pipeline_config["mlflow_experiment"])

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