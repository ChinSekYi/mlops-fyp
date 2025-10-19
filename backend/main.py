"""
Fraud Detection API using FastAPI, MLflow, and a registered ML model.
Provides endpoints for health check, prediction, model info, metrics,
and feature info.
"""

import os

import mlflow
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel

from backend.utils import (
    get_model_version_details,
    get_run_details,
    load_environment,
    load_model_and_preprocessor,
    predict,
)
from src.pipeline.predict_pipeline import CustomData

env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model, preprocessor = load_model_and_preprocessor(MODEL_NAME, MODEL_ALIAS)
app = FastAPI(title="Fraud Detection API", version="1.0")


class TransactionInput(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str
    nameOrig: str
    nameDest: str


@app.get("/")
def health_check():
    """Health check endpoint to verify API status and model alias."""
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS}


@app.post("/predict")
def predict_endpoint(inputData: TransactionInput):
    """Predicts whether a transaction is fraudulent (1) or not (0) using the trained model."""
    custom_data = CustomData(**inputData.dict())
    return predict(custom_data, model, preprocessor)


@app.get("/model-info")
def get_model_info():
    """Returns metadata about the currently loaded ML model from MLflow."""
    try:
        client = MlflowClient()
        model_version_details = get_model_version_details(
            client, MODEL_NAME, MODEL_ALIAS
        )
        return {
            "model_name": model_version_details.name,
            "version": model_version_details.version,
            "alias": MODEL_ALIAS,
            "run_id": model_version_details.run_id,
            "status": model_version_details.status,
            "creation_timestamp": model_version_details.creation_timestamp,
            "source": "mlflow_alias",
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "environment": os.getenv("ENV_USED", "unknown"),
        }
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        return {"error": str(e)}


@app.get("/metrics")
def get_model_metrics():
    """Returns metrics and parameters for the current model run from MLflow."""
    try:
        client = MlflowClient()
        model_version_details = get_model_version_details(
            client, MODEL_NAME, MODEL_ALIAS
        )
        run = get_run_details(client, model_version_details.run_id)
        return {
            "run_id": model_version_details.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        return {"error": str(e)}


@app.get("/feature-info")
def feature_info():
    """Returns information about the model's input features for PaySim."""
    return {
        "expected_features": CustomData.EXPECTED_FEATURES,
        "type_dropdown_values": ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        "description": "PaySim transaction features. 'type' should be selected from the dropdown.",
        "example": {
            "step": 1,
            "amount": 1000.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "type": "CASH_OUT",
            "nameOrig": "C84071102",
            "nameDest": "C1576697216",
        },
    }


if __name__ == "__main__":
    pass
    # sample curl request
    """
        curl -X POST "http://localhost:8001/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "step": 1,
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0,
        "type": "CASH_OUT",
        "nameOrig": "C84071102",
        "nameDest": "C1576697216"
      }'
    """
