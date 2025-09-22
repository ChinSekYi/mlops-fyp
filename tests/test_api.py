import os
import mlflow
import requests
from dotenv import load_dotenv
from api.main import load_model
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") #overrided by ENV in docker-compose.yml
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")  
MODEL_ALIAS = os.getenv("MODEL_ALIAS")

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def test_model_load():
    model = load_model()
    assert model is not None
    
    # Predict on a sample row
    sample = [[0.0]*30]
    preds = model.predict(sample)
    assert len(preds) == 1
