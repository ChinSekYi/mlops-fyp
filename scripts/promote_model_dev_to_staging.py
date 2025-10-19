import os
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv


def load_environment(env_file: str = None):
    load_dotenv(env_file or ".env")

load_environment(os.getenv("ENV_FILE", ".env"))
MLFLOW_TRACKING_PRIVATE_IP_DEV = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_DEV")
MLFLOW_TRACKING_PRIVATE_IP_STAGING = os.getenv("MLFLOW_TRACKING_PRIVATE_IP_STAGING")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")

# Configuration
DEV_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_DEV}:5050"
STAGING_TRACKING_URI = f"http://{MLFLOW_TRACKING_PRIVATE_IP_STAGING}:5050"
MODEL_ALIAS = "champion"
STAGING_MODEL_NAME = "staging.fraud-detection-model"  # Or use same name if desired

# --- STEP 1: Download model from DEV ---
mlflow.set_tracking_uri(DEV_TRACKING_URI)
dev_client = MlflowClient()
model_version = dev_client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
run_id = model_version.run_id

# Download model artifact locally
model_uri = f"models:/{MODEL_NAME}/{model_version}"
dev_chamption_model = mlflow.sklearn.load_model(model_uri)
local_path = f"../artifacts/model/"
print(f"Downloaded model to {local_path}")

# Retrieve model_name from run tags
run_info = dev_client.get_run(run_id)
model_name_tag = run_info.data.tags.get("model_name", "champion_model")

# --- STEP 2: Register model to STAGING ---
mlflow.set_tracking_uri(STAGING_TRACKING_URI)
staging_client = MlflowClient()

with mlflow.start_run(run_name="model_promotion") as run:
    mlflow.sklearn.log_model(
        sk_model=mlflow.sklearn.load_model(local_path),
        name=model_name_tag,
        registered_model_name=STAGING_MODEL_NAME,
    )

    # Model registry and version creation is handled by mlflow.sklearn.log_model
    print(f"Promoted model to staging: {STAGING_MODEL_NAME}")
    