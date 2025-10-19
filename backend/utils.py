import os

import joblib
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

from src.pipeline.predict_pipeline import CustomData, DummyModel, PredictPipeline


def load_environment(env_file: str = None):
    """
    Load environment variables from a file in /env directory.
    Default is .env at project root's /env folder.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    env_dir = os.path.join(project_root, "env")
    if env_file is None:
        env_file = "env/.env"
    env_path = (
        os.path.join(project_root, env_file)
        if not os.path.isabs(env_file)
        else env_file
    )
    if not os.path.exists(env_path):
        env_path = os.path.join(env_dir, ".env")
    load_dotenv(env_path)
    print(f"[ENV] Loaded environment from {env_path}")


def load_model_and_preprocessor(MODEL_NAME, MODEL_ALIAS):
    """Loads both the ML model and preprocessor from MLflow or local files."""
    client = MlflowClient()
    try:
        model_version = get_model_version_details(client, MODEL_NAME, MODEL_ALIAS)
        version = model_version.version
        run_id = model_version.run_id
        print(f"Attempting to load model {MODEL_NAME} version {version} from MLflow")
        model_uri = f"models:/{MODEL_NAME}/{version}"
        model = mlflow.sklearn.load_model(model_uri)

        # Try to load preprocessor from MLflow artifacts
        try:
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
        print(f"Could not load MLflow model or preprocessor: {e}")
        print("Using dummy model for predictions")
        return DummyModel(), None


def predict(custom_data: CustomData, model, preprocessor):
    input_df = custom_data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred_result = predict_pipeline.predict(input_df, model, preprocessor, DummyModel)
    return {"prediction": int(pred_result[0])}


def get_model_version_details(client, model_name, model_alias):
    """Get model version details by alias, fallback to version 1."""
    try:
        return client.get_model_version_by_alias(name=model_name, alias=model_alias)
    except Exception:
        return client.get_model_version(model_name, "1")


def get_run_details(client, run_id):
    """Get run details for a given run_id."""
    return client.get_run(run_id)


if __name__ == "__main__":
    # Example test for predict()

    # Load environment variables
    load_environment()
    MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS")

    model, preprocessor = load_model_and_preprocessor(MODEL_NAME, MODEL_ALIAS)

    sample = {
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
    custom_data = CustomData(**sample)
    result = predict(custom_data, model, preprocessor)
    print("Prediction result:", result)
