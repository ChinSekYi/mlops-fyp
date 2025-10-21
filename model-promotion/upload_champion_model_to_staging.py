"""
Script: upload_champion_model_to_staging.py

Uploads a locally saved champion model to the staging MLflow server and registers it.
Use AWS_PROFILE=stag-bkt and ENV_FILE=.env.stag_machine.

REMINDER:
Only use this script with stag-bkt credentials.
Run with:
    AWS_PROFILE=stag-bkt ENV_FILE=.env.stag_machine
"""

import json
import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature

STAGING_MODEL_NAME = "staging.fraud-detection-model"
LOCAL_MODEL_DIR = "artifacts/models/dev_champion_model"


def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")


# Load env and configurations
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
    if signature:
        signature = ModelSignature.from_dict(signature)
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

    # Log preprocessor artifact from fixed path (as in model_trainer)
    preprocessor_path = "artifacts/preprocessor/preprocessor.pkl"
    if os.path.isfile(preprocessor_path):
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        print(f"Logged preprocessor artifact: {preprocessor_path}")
    else:
        print(f"No preprocessor artifact found at {preprocessor_path}")

    print(f"Promoted model to staging: {STAGING_MODEL_NAME}")
