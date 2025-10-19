import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient


def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")


load_environment(os.getenv("ENV_FILE", ".env"))
MLFLOW_TRACKING_PRIVATE_IP_STAGING = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_STAGING")
MLFLOW_TRACKING_PRIVATE_IP_PROD = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_PROD")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")

# Configuration
STAGING_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_STAGING}:5050"
PROD_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_PROD}:5050"
MODEL_ALIAS = "champion"
PROD_MODEL_NAME = "prod.fraud-detection-model"  # Or use same name if desired

# --- STEP 1: Download model from STAGING ---
mlflow.set_tracking_uri(STAGING_TRACKING_URI)
staging_client = MlflowClient()
model_version = staging_client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
run_id = model_version.run_id

# Download model artifact locally
model_uri = f"models:/{MODEL_NAME}/{model_version}"
staging_champion_model = mlflow.sklearn.load_model(model_uri)
local_path = f"../artifacts/model/"
print(f"Downloaded model to {local_path}")

# Retrieve model_name from run tags
run_info = staging_client.get_run(run_id)
model_name_tag = run_info.data.tags.get("model_name", "champion_model")

# --- STEP 2: Register model to PROD ---
mlflow.set_tracking_uri(PROD_TRACKING_URI)
prod_client = MlflowClient()

with mlflow.start_run(run_name="model_promotion") as run:
    mlflow.sklearn.log_model(
        sk_model=mlflow.sklearn.load_model(local_path),
        name=model_name_tag,
        registered_model_name=PROD_MODEL_NAME,
    )
    # Model registry creation and versioning handled by MLflow
    print(f"Promoted model to prod: {PROD_MODEL_NAME}")
