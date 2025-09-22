import os
import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
#MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")  
MLFLOW_TRACKING_URI = 'http://mlflow-server:5050'
MODEL_NAME='fraud-detection-model'
MODEL_ALIAS = "champion"      # e.g. "champion", "staging"

# FastAPI App
app = FastAPI(title="Fraud Detection API", version="1.0")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from {model_uri}")
    return mlflow.sklearn.load_model(model_uri)

model = load_model()

# Input Schema - ensures validation
class Transaction(BaseModel):
    features: list[float]   # e.g., 30 numerical features for credit card fraud

# Routes
@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert features into DataFrame
    df = pd.DataFrame([transaction.features])
    preds = model.predict(df)
    return {"prediction": int(preds[0])}   # binary classification: 0 or 1

class BatchTransaction(BaseModel):
    features_list: list[list[float]]   # list of multiple feature vectors

@app.post("/batch-predict")
def batch_predict(batch: BatchTransaction):
    df = pd.DataFrame(batch.features_list)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}

@app.get("/model-info")
def get_model_info():
    client = MlflowClient()

    model_version_details = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS
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
    client = MlflowClient()
    model_version_details = client.get_model_version_by_alias(
        name=MODEL_NAME, alias=MODEL_ALIAS
    )
    run_id = model_version_details.run_id
    run = client.get_run(run_id)

    return {
        "run_id": run_id,
        "metrics": run.data.metrics,   # accuracy, auc, f1, etc.
        "params": run.data.params      # hyperparameters
    }

@app.get("/feature-info")
def feature_info():
    return {
        "num_features": 30,
        "description": "PCA-transformed credit card transaction features",
        "source": "Kaggle Credit Card Fraud Dataset"
    }

if __name__ == "__main__":
    print(MODEL_NAME)
    print(model)