import os
import mlflow
from mlflow import MlflowClient
from typing import Optional, Dict


def get_client(tracking_uri: Optional[str] = "http://127.0.0.1:8080") -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)

def get_experiments(client):
    return client.search_experiments()

def create_experiment(client: MlflowClient, name: str, description: str = "", tags: Dict = None) -> str:
    if tags is None:
        tags = {}
    tags["mlflow.note.content"] = description
    experiment_id = client.create_experiment(name=name, tags=tags)
    return experiment_id

def set_experiment(experiment_name: str, tracking_uri: str = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def search_experiment(client: MlflowClient, tag_key: str, tag_value: str):
    filter_str = f"tags.`{tag_key}` = '{tag_value}'"
    return client.search_experiments(filter_string=filter_str)

def start_run(run_name: str = None):
    return mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    mlflow.log_params(params)

def log_metrics(metrics: dict, step: int = None):
    mlflow.log_metrics(metrics, step=step)

def log_artifact(file_path: str, artifact_path: str = None):
    mlflow.log_artifact(file_path, artifact_path)

def log_model(model, artifact_path: str = "models"):
    mlflow.sklearn.log_model(model, artifact_path)