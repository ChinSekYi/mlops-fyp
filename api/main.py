import os
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.pipeline.predict_pipeline import CustomData

# Config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")  
MODEL_ALIAS = "champion"      # e.g. "champion", "staging"

# FastAPI App
app = FastAPI(title="Fraud Detection API", version="1.0")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)

model = load_model()

# Input Schema
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
    preds = model.predict(df)
    return {"prediction": int(preds[0])}   # binary classification: 0 or 1
