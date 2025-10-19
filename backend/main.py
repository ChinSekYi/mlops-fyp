"""
Fraud Detection API using FastAPI, MLflow, and a registered ML model.
Provides endpoints for health check, prediction, model info, metrics,
and feature info.
"""

import os

import joblib
import mlflow
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel

from backend.utils import load_environment
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Fraud Detection API", version="1.0")

EXPECTED_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",
    "nameOrig",
    "nameDest",
]


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
        # Try to load model using alias first
        print(f"Attempting to load model {MODEL_NAME} with alias {MODEL_ALIAS}")
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        print("Successfully loaded MLflow model")

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


@app.get("/")
def health_check():
    """Health check endpoint to verify API status and model alias."""
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS}


@app.post("/predict")
def predict(inputData: TransactionInput):
    """Predicts whether a transaction is fraudulent (1) or not (0) using the trained model."""
    custom_data = CustomData(**inputData.dict())
    input_df = custom_data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred_result = predict_pipeline.predict(input_df)
    return {"prediction": int(pred_result[0])}


@app.get("/model-info")
def get_model_info():
    """Returns metadata about the currently loaded ML model from MLflow."""
    try:
        client = MlflowClient()

        # Try to get model info by alias first
        try:
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
                "source": "mlflow_alias",
                "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
                "environment": os.getenv("ENV_USED", "unknown"),
            }

        except Exception as alias_error:
            print(f"Could not get model info by alias: {alias_error}")

            # Fallback to version 1
            try:
                model_version_details = client.get_model_version(MODEL_NAME, "1")

                return {
                    "model_name": model_version_details.name,
                    "version": model_version_details.version,
                    "alias": "none (using version 1)",
                    "run_id": model_version_details.run_id,
                    "status": model_version_details.status,
                    "creation_timestamp": model_version_details.creation_timestamp,
                    "source": "mlflow_version_1",
                    "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
                    "s3_bucket": os.getenv("MLFLOW_S3_BUCKET"),
                    "environment": os.getenv("ENV_USED", "unknown"),
                }

            except Exception as version_error:
                print(f"Could not get model info by version: {version_error}")
                raise version_error

    except Exception as e:
        print(f"MLflow connection failed: {e}")
        # Return basic info about the current setup
        return {
            "model_name": MODEL_NAME,
            "version": "unknown",
            "alias": MODEL_ALIAS,
            "run_id": "unknown",
            "status": "using_dummy_model",
            "creation_timestamp": "unknown",
            "source": "dummy_fallback",
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "s3_bucket": os.getenv("MLFLOW_S3_BUCKET"),
            "environment": os.getenv("ENV_USED", "unknown"),
            "error": str(e),
        }


@app.get("/metrics")
def get_model_metrics():
    """Returns metrics and parameters for the current model run from MLflow."""
    try:
        client = MlflowClient()

        # Try to get model version by alias first
        try:
            model_version_details = client.get_model_version_by_alias(
                name=MODEL_NAME, alias=MODEL_ALIAS
            )
            run_id = model_version_details.run_id

        except Exception as alias_error:
            print(f"Could not get model version by alias: {alias_error}")

            # Fallback to version 1
            try:
                model_version_details = client.get_model_version(MODEL_NAME, "1")
                run_id = model_version_details.run_id

            except Exception as version_error:
                print(f"Could not get model version: {version_error}")
                raise version_error

        run = client.get_run(run_id)

        return {
            "run_id": run_id,
            "metrics": run.data.metrics,  # accuracy, auc, f1, etc.
            "params": run.data.params,  # hyperparameters
        }

    except Exception as e:
        print(f"MLflow connection failed: {e}")
        return {
            "run_id": "unknown",
            "metrics": {"error": "MLflow unavailable"},
            "params": {"error": "MLflow unavailable"},
            "error": str(e),
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
    pass
    # sample curl request
    """
        curl -X POST "http://localhost:8000/predict" \
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
