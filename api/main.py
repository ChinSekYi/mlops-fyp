"""
Fraud Detection API using FastAPI, MLflow, and a registered ML model.
Provides endpoints for health check, prediction, model info, metrics,
and feature info.
"""

import os

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from api.utils import load_environment

env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Fraud Detection API", version="1.0")

# Test render connectivity to MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print("Active experiment:", mlflow.get_tracking_uri())


class DummyModel:
    # Dummy model for when MLflow is not available

    def predict(self, X):
        # Simple heuristic: classify as fraud if amount > 200000 or oldbalanceOrg == 0
        import numpy as np

        predictions = []
        for _, row in X.iterrows():
            if row["amount"] > 200000 or (
                row["oldbalanceOrg"] == 0 and row["amount"] > 0
            ):
                predictions.append(1)  # Fraud
            else:
                predictions.append(0)  # Not fraud
        return np.array(predictions)


def load_model_and_preprocessor():
    """Loads both the ML model and preprocessor from MLflow or local files."""
    try:
        print(f"Attempting to load model {MODEL_NAME} with alias {MODEL_ALIAS}")
        # Try to load model using alias first
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Successfully loaded MLflow model")

        # Try to load preprocessor from artifacts
        try:
            # Get the run_id from the model version
            client = MlflowClient()
            model_version = client.get_model_version_by_alias(
                name=MODEL_NAME, alias=MODEL_ALIAS
            )
            run_id = model_version.run_id

            # Download preprocessor artifact (it's in the 'preprocessor' directory)
            preprocessor_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="preprocessor"
            )

            preprocessor_files = os.listdir(preprocessor_path)
            preprocessor_file = None
            for file in preprocessor_files:
                if file.endswith(".pkl"):
                    preprocessor_file = os.path.join(preprocessor_path, file)
                    break

            if preprocessor_file:
                preprocessor = joblib.load(preprocessor_file)
                print("Successfully loaded preprocessor from MLflow")
            else:
                print("No .pkl file found in preprocessor artifacts")
                preprocessor = None

        except Exception as e:
            print(f"Could not load preprocessor from MLflow: {e}")
            preprocessor = None

        return model, preprocessor

    except Exception as e:
        print(f"Could not load MLflow model: {e}")
        try:
            # Fallback to version 1 if alias doesn't exist
            model_version = 1
            model_uri = f"models:/{MODEL_NAME}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Successfully loaded MLflow model version {model_version}")
            return model, None
        except Exception as e2:
            print(f"Could not load any MLflow model: {e2}")
            print("Using dummy model for predictions")
            return DummyModel(), None


model, preprocessor = load_model_and_preprocessor()

EXPECTED_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",  # Raw categorical input (CASH_IN, CASH_OUT, etc.)
    "nameOrig",  # Raw string input (e.g., "C84071102")
    "nameDest",  # Raw string input (e.g., "C1576697216")
]


class Transaction(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str = Field(..., description="Transaction type", example="CASH_OUT")
    nameOrig: str = Field(..., description="Origin account name", example="C84071102")
    nameDest: str = Field(
        ..., description="Destination account name", example="C1576697216"
    )


@app.get("/")
def health_check():
    """Health check endpoint to verify API status and model alias."""
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS}


@app.post("/predict")
def predict(transaction: Transaction):
    """Predicts whether a transaction is fraudulent (1) or not (0) using the trained model."""

    # Convert input to DataFrame
    input_dict = transaction.model_dump()
    df = pd.DataFrame([input_dict])

    # If we have a preprocessor, use it (but need to tokenize first)
    if preprocessor is not None:
        try:
            # Apply tokenization first (same as in training)
            from src.utils import tokenize_column

            # Create dummy test dataframe for tokenization (tokenize_column expects train and test)
            df_dummy = df.copy()
            df_tokenized, _ = tokenize_column(df, df_dummy, "nameOrig")
            df_tokenized, _ = tokenize_column(df_tokenized, df_dummy, "nameDest")

            # Drop original name columns
            df_tokenized = df_tokenized.drop(["nameOrig", "nameDest"], axis=1)

            # Apply the preprocessor
            processed_features = preprocessor.transform(df_tokenized)
            preds = model.predict(processed_features)
            return {"prediction": int(preds[0])}
        except Exception as e:
            print(f"⚠️ Preprocessor failed, falling back to manual preprocessing: {e}")

    # Fallback: Manual preprocessing (for when preprocessor not available)
    # Tokenize names manually
    df["nameOrig_token"] = df["nameOrig"].str.extract(r"(\d+)").astype(int)
    df["nameDest_token"] = df["nameDest"].str.extract(r"(\d+)").astype(int)
    df = df.drop(["nameOrig", "nameDest"], axis=1)

    # One-hot encode 'type'
    type_values = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
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
            "nameOrig": "C84071102",
            "nameDest": "C1576697216",
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
            "nameOrig": "C84071102",
            "nameDest": "C1576697216",
        }
    )
    output = predict(input)
    print(output)
