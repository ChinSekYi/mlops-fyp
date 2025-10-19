import json
import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

"""
Script: upload_champion_model_to_staging.py

Uploads a locally saved champion model to the staging MLflow server and registers it.
Use AWS_PROFILE=stag-bkt and ENV_FILE=env/.env.stag_machine.

REMINDER:
Only use this script with stag-bkt credentials.
Run with:
    AWS_PROFILE=stag-bkt ENV_FILE=env/.env.stag_machine python3 scripts/upload_champion_model_to_staging.py
"""
STAGING_MODEL_NAME = "staging.fraud-detection-model"  # Or use same name if desired
LOCAL_MODEL_DIR = "artifacts/models/dev_champion_model"


# Use MLFLOW_TRACKING_URI from environment (should be set in .env.stag_machine)
def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")


load_environment(os.getenv("ENV_FILE", "env/.env.stag_machine"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
staging_client = MlflowClient()

# Load run metadata
metadata_path = os.path.join(LOCAL_MODEL_DIR, "run_metadata.json")
with open(metadata_path, "r") as f:
    metadata = json.load(f)

model_name_tag = "champion_model"

with mlflow.start_run(run_name="model_promotion") as run:
    # Log metrics, params, tags
    for k, v in metadata.get("metrics", {}).items():
        mlflow.log_metric(k, v)
    for k, v in metadata.get("params", {}).items():
        mlflow.log_param(k, v)
    for k, v in metadata.get("tags", {}).items():
        mlflow.set_tag(k, v)

    # Log model with signature and input_example if available
    signature = metadata.get("signature")
    input_example = metadata.get("input_example")
    log_model_kwargs = dict(
        sk_model=mlflow.sklearn.load_model(LOCAL_MODEL_DIR),
        name=model_name_tag,
        registered_model_name=STAGING_MODEL_NAME,
    )
    if signature:
        log_model_kwargs["signature"] = signature
    if input_example:
        log_model_kwargs["input_example"] = input_example
    mlflow.sklearn.log_model(**log_model_kwargs)

    # Log preprocessor artifact if exists
    preprocessor_dir = os.path.join(LOCAL_MODEL_DIR, "preprocessor")
    if os.path.exists(preprocessor_dir):
        for file in os.listdir(preprocessor_dir):
            file_path = os.path.join(preprocessor_dir, file)
            if os.path.isfile(file_path):
                mlflow.log_artifact(file_path, artifact_path="preprocessor")
        print(f"Logged preprocessor artifact from {preprocessor_dir}")
    else:
        print("No preprocessor artifact found to log.")

    print(f"Promoted model to staging: {STAGING_MODEL_NAME}")
