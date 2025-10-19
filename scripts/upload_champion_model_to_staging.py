"""
Script: upload_champion_model_to_staging.py

Uploads a locally saved champion model to the staging MLflow server and registers it.
Use AWS_PROFILE=stag-bkt and ENV_FILE=env/.env.stag_machine.

REMINDER:
- Only use this script with stag-bkt credentials.
- Run with:
    AWS_PROFILE=stag-bkt ENV_FILE=env/.env.stag_machine python3 scripts/upload_champion_model_to_staging.py
"""
import os
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")

load_environment(os.getenv("ENV_FILE", "env/.env.stag_machine"))
MLFLOW_TRACKING_PRIVATE_IP_STAGING = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_STAGING")
STAGING_MODEL_NAME = "staging.fraud-detection-model"  # Or use same name if desired
LOCAL_MODEL_DIR = "artifacts/models/dev_champion_model/model"

STAGING_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_STAGING}:5050"
mlflow.set_tracking_uri(STAGING_TRACKING_URI)
staging_client = MlflowClient()

model_name_tag = "champion_model" 

with mlflow.start_run(run_name="model_promotion") as run:
    mlflow.sklearn.log_model(
        sk_model=mlflow.sklearn.load_model(LOCAL_MODEL_DIR),
        name=model_name_tag,
        registered_model_name=STAGING_MODEL_NAME,
    )
    print(f"Promoted model to staging: {STAGING_MODEL_NAME}")
