"""
Script: download_champion_model_staging.py

Downloads the champion model from the staging MLflow server and saves it locally as an MLflow model directory.
Use AWS_PROFILE=stag-bkt and ENV_FILE=env/.env.stag_machine.

REMINDER:
- Only use this script with stag-bkt credentials.
- Run with:
    AWS_PROFILE=stag-bkt ENV_FILE=env/.env.stag_machine python3 scripts/download_champion_model_staging.py
"""
import os
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")

load_environment(os.getenv("ENV_FILE", "env/.env.stag_machine"))
MLFLOW_TRACKING_PRIVATE_IP_STAGING = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_STAGING")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = "champion"

STAGING_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_STAGING}:5050"
LOCAL_MODEL_DIR = "artifacts/models/staging_champion_model"

mlflow.set_tracking_uri(STAGING_TRACKING_URI)
staging_client = MlflowClient()
model_version = staging_client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
run_id = model_version.run_id
print(f"Champion staging model is version {model_version.version}")

model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
print(f"Downloading model from {model_uri} to {LOCAL_MODEL_DIR}")
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=LOCAL_MODEL_DIR)
print(f"Model downloaded to {LOCAL_MODEL_DIR}")
