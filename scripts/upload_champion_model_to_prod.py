"""
Script: upload_champion_model_to_prod.py

Uploads a locally saved champion model to the prod MLflow server and registers it.
Use AWS_PROFILE=prod-bkt and ENV_FILE=env/.env.prod_machine.

REMINDER:
- Only use this script with prod-bkt credentials.
- Run with:
    AWS_PROFILE=prod-bkt ENV_FILE=env/.env.prod_machine python3 scripts/upload_champion_model_to_prod.py
"""
import os
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")

load_environment(os.getenv("ENV_FILE", "env/.env.prod_machine"))
MLFLOW_TRACKING_PRIVATE_IP_PROD = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_PROD")
PROD_MODEL_NAME = "prod.fraud-detection-model"  # Or use same name if desired
LOCAL_MODEL_DIR = "artifacts/models/staging_champion_model/model"

PROD_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_PROD}:5050"
mlflow.set_tracking_uri(PROD_TRACKING_URI)
prod_client = MlflowClient()

# Load model_name_tag from a metadata file if you saved it, or set manually
model_name_tag = "champion_model" 

with mlflow.start_run(run_name="model_promotion") as run:
    mlflow.sklearn.log_model(
        sk_model=mlflow.sklearn.load_model(LOCAL_MODEL_DIR),
        name=model_name_tag,
        registered_model_name=PROD_MODEL_NAME,
    )
    print(f"Promoted model to prod: {PROD_MODEL_NAME}")
