import mlflow
import os
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion

# Set MLflow tracking URI to local MLflow server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT"))

# Start MLflow run
with mlflow.start_run(run_name="logistic_regression_test_13"):
    mlflow.set_tag("team", "pacman")
    dataset_name = "creditcard.csv"
    
    # ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion(dataset_name)

    # transformation
    data_transformation = DataTransformation()
    train_arr_path, test_arr_path, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    # training
    modeltrainer = ModelTrainer()
    artifact_path = "logistic_regression_model"
    training_metrics, testing_metrics = modeltrainer.initiate_model_trainer(train_arr_path, test_arr_path, artifact_path)