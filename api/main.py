"""
Fraud Detection API using FastAPI, MLflow, and a registered ML model.
Provides endpoints for health check, prediction, model info, metrics,
and feature info.
"""

import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Fraud Detection API", version="1.0")


def load_model():
    """Loads the ML model from MLflow using the registered model name and alias."""

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


model = load_model()

EXPECTED_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",
    "nameOrig_token",
    "nameDest_token",
]


class Transaction(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str = Field(..., description="Transaction type", example="CASH_OUT")
    nameOrig_token: int
    nameDest_token: int


@app.get("/")
def health_check():
    """Health check endpoint to verify API status and model alias."""
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS}


@app.post("/predict")
def predict(transaction: Transaction):
    """Predicts whether a transaction is fraudulent (1) or not (0) using the trained model."""

    # One-hot encode 'type'
    type_values = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    input_dict = transaction.model_dump()
    df = pd.DataFrame([input_dict])

    for t in type_values:
        df[f"type__{t}"] = (df["type"] == t).astype(int)
    df = df.drop(columns=["type"])

    ordered_cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type__CASH_IN",
        "type__CASH_OUT",
        "type__DEBIT",
        "type__PAYMENT",
        "type__TRANSFER",
        "nameOrig_token",
        "nameDest_token",
    ]
    df = df[ordered_cols]
    preds = model.predict(df)
    return {"prediction": int(preds[0])}


@app.get("/model-info")
def get_model_info():
    """Returns metadata about the currently loaded ML model from MLflow."""
    client = MlflowClient()

    model_version_details = client.get_model_version_by_alias(
        name=MODEL_NAME, alias=MODEL_ALIAS
    )

    return {
        "model_name": model_version_details.name,
        "version": model_version_details.version,
        "alias": MODEL_ALIAS,
        "run_id": model_version_details.run_id,
        "status": model_version_details.status,
        "creation_timestamp": model_version_details.creation_timestamp,
        "current_stage": model_version_details.current_stage,
        "source": model_version_details.source,
    }


@app.get("/metrics")
def get_model_metrics():
    """Returns metrics and parameters for the current model run from MLflow."""

    client = MlflowClient()
    model_version_details = client.get_model_version_by_alias(
        name=MODEL_NAME, alias=MODEL_ALIAS
    )
    run_id = model_version_details.run_id
    run = client.get_run(run_id)

    return {
        "run_id": run_id,
        "metrics": run.data.metrics,  # accuracy, auc, f1, etc.
        "params": run.data.params,  # hyperparameters
    }


@app.get("/feature-info")
def feature_info():
    """Returns information about the model's input features for PaySim."""
    return {
        "expected_features": EXPECTED_FEATURES,
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
            "nameOrig_token": 123,
            "nameDest_token": 456,
        },
    }


if __name__ == "__main__":
    print(MODEL_NAME)
    print(model)

    input = Transaction(
        **{
            "step": 1,
            "amount": 1000.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "type": "CASH_OUT",
            "nameOrig_token": 123,
            "nameDest_token": 456,
        }
    )
    output = predict(input)
    print(output)
