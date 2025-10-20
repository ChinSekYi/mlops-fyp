"""
Script: download_champion_model_dev.py

Downloads the champion model from the dev MLflow server and saves it locally as an MLflow model directory.
Use AWS_PROFILE=dev-bkt and ENV_FILE=env/.env.dev_machine.

REMINDER:
- Only use this script with dev-bkt credentials.
- Run with:
    AWS_PROFILE=dev-bkt ENV_FILE=env/.env.dev_machine python3 scripts/download_champion_model_dev.py
"""

import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient


def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")


load_environment(os.getenv("ENV_FILE", "env/.env.dev_machine"))
MLFLOW_TRACKING_PRIVATE_IP_DEV = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_DEV")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = "champion"

DEV_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_DEV}:5050"
LOCAL_MODEL_DIR = "artifacts/models/dev_champion_model"


mlflow.set_tracking_uri(DEV_TRACKING_URI)
dev_client = MlflowClient()
model_version = dev_client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
run_id = model_version.run_id
print(f"Champion dev model is version {model_version.version}")

# Download model directory
model_uri = f"models:/{MODEL_NAME}/{model_version.version}"
print(f"Downloading model from {model_uri} to {LOCAL_MODEL_DIR}")
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=LOCAL_MODEL_DIR)
print(f"Model downloaded to {LOCAL_MODEL_DIR}")

# Download preprocessor artifact if exists
preprocessor_artifact_path = f"runs:/{run_id}/preprocessor"
preprocessor_local_dir = os.path.join(LOCAL_MODEL_DIR, "preprocessor")
try:
    mlflow.artifacts.download_artifacts(
        artifact_uri=preprocessor_artifact_path, dst_path=preprocessor_local_dir
    )
    print(f"Preprocessor downloaded to {preprocessor_local_dir}")
except Exception as e:
    print(f"No preprocessor artifact found: {e}")

# Save run metadata (metrics, params, tags, signature, input_example)
import json

run_info = dev_client.get_run(run_id)
metadata = {
    "metrics": run_info.data.metrics,
    "params": run_info.data.params,
    "tags": run_info.data.tags,
}
# Try to get signature and input_example from MLmodel file
mlmodel_path = os.path.join(LOCAL_MODEL_DIR, "MLmodel")
signature = None
input_example = None
try:
    import yaml

    with open(mlmodel_path, "r") as f:
        mlmodel_yaml = yaml.safe_load(f)
    signature = mlmodel_yaml.get("signature")
    input_example = mlmodel_yaml.get("input_example")
except Exception as e:
    print(f"Could not parse MLmodel for signature/input_example: {e}")
metadata["signature"] = signature
metadata["input_example"] = input_example

with open(os.path.join(LOCAL_MODEL_DIR, "run_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved run metadata to {os.path.join(LOCAL_MODEL_DIR, 'run_metadata.json')}")
