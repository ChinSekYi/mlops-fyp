# pylint: skip-file
# flake8: noqa

import os
import sys

import mlflow
import numpy as np
import pandas as pd

from src.core.exception import CustomException
from src.core.utils import load_config, load_environment, load_object, tokenize_column

env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

config = load_config()

predict_config = config["predict_pipeline"]
model_path = predict_config["model_path"]
preprocessor_path = predict_config["preprocessor_path"]


class DummyModel:

    def predict(self, X):
        # Simple heuristic: classify as fraud if amount > 200000 or oldbalanceOrg == 0
        predictions = []
        for _, row in X.iterrows():
            if row["amount"] > 200000 or (
                row["oldbalanceOrg"] == 0 and row["amount"] > 0
            ):
                predictions.append(1)  # Fraud
            else:
                predictions.append(0)  # Not fraud
        return np.array(predictions)


class PredictPipeline:

    def predict(self, features, model, preprocessor, DummyModel):
        """
        Makes a prediction using the loaded ML model and preprocessor.
        If MLflow model or preprocessor fails, falls back to DummyModel or raw features.
        """
        try:
            if model is None:
                raise ValueError("Model could not be loaded.")
            if preprocessor is not None:
                try:
                    data_scaled = preprocessor.transform(features)
                    pred_result = model.predict(data_scaled)
                except Exception as preproc_error:
                    print(
                        f"Preprocessor transform failed: {preproc_error}. Using raw features."
                    )
                    pred_result = model.predict(features)
            else:
                print("Preprocessor not found. Using raw features.")
                pred_result = model.predict(features)
            return pred_result
        except Exception as e:
            print(f"Prediction failed: {e}. Falling back to DummyModel.")
            dummy_model = DummyModel()
            try:
                pred_result = dummy_model.predict(features)
                return pred_result
            except Exception as dummy_error:
                print(f"DummyModel prediction failed: {dummy_error}")
                raise CustomException(dummy_error, sys)


class CustomData:
    """
    Responsible for taking PaySim transaction inputs (RAW FORMAT with string account names)
    and converting them to the format expected by the fraud detection model.
    """

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

    def __init__(
        self,
        step: int,
        amount: float,
        oldbalanceOrg: float,
        newbalanceOrig: float,
        oldbalanceDest: float,
        newbalanceDest: float,
        type: str,  # Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
        nameOrig: str,  # Raw account ID like "C1900756070"
        nameDest: str,  # Raw account ID like "C1995455020"
    ):
        self.step = step
        self.amount = amount
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig
        self.oldbalanceDest = oldbalanceDest
        self.newbalanceDest = newbalanceDest
        self.type = type
        self.nameOrig = nameOrig
        self.nameDest = nameDest

    def get_data_as_dataframe(self):
        try:
            # Step 1: Create dataframe with raw data
            custom_data_input = {
                "step": [self.step],
                "amount": [self.amount],
                "oldbalanceOrg": [self.oldbalanceOrg],
                "newbalanceOrig": [self.newbalanceOrig],
                "oldbalanceDest": [self.oldbalanceDest],
                "newbalanceDest": [self.newbalanceDest],
                "type": [self.type],
                "nameOrig": [self.nameOrig],
                "nameDest": [self.nameDest],
            }
            df = pd.DataFrame(custom_data_input)

            # Step 2: Tokenize account names (same approach as API)
            try:
                df_dummy = df.copy()
                df_tokenized, _ = tokenize_column(df, df_dummy, "nameOrig")
                df_tokenized, _ = tokenize_column(df_tokenized, df_dummy, "nameDest")
                df = df_tokenized
            except Exception as tokenize_error:
                print(
                    f"Tokenization function failed, using manual tokenization: {tokenize_error}"
                )
                # Fallback: Manual tokenization (extract numbers from account IDs)
                df["nameOrig_token"] = df["nameOrig"].str.extract(r"(\d+)").astype(int)
                df["nameDest_token"] = df["nameDest"].str.extract(r"(\d+)").astype(int)

            # Step 3: Drop raw account columns (preprocessor doesn't expect them)
            df = df.drop(columns=["nameOrig", "nameDest"])

            # Step 4: Reorder columns to match what preprocessor expects
            ordered_cols = [
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "nameOrig_token",
                "nameDest_token",
                "type",  # Categorical feature - preprocessor will one-hot encode this
            ]
            df = df[ordered_cols]

            return df
        except Exception as e:
            import sys

            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test with both non-fraud and fraud examples using RAW data format

    print("=" * 60)
    print("Testing Prediction Pipeline with Raw PaySim Data")
    print("=" * 60)

    # use local model and preprocessor
    model = load_object(file_path=model_path)
    preprocessor = load_object(file_path=preprocessor_path)

    # Non-fraud example
    print("\n1. Testing Non-Fraud Example:")
    data_nonfraud = CustomData(
        step=1,
        amount=1000.0,
        oldbalanceOrg=5000.0,
        newbalanceOrig=4000.0,
        oldbalanceDest=0.0,
        newbalanceDest=1000.0,
        type="CASH_OUT",
        nameOrig="C84071102",  # Raw account ID
        nameDest="C1576697216",  # Raw account ID
    )

    pred_df_nonfraud = data_nonfraud.get_data_as_dataframe()
    print("Input DataFrame (after tokenization):")
    print(pred_df_nonfraud)
    print(f"DataFrame shape: {pred_df_nonfraud.shape}")
    print(f"Columns: {list(pred_df_nonfraud.columns)}")

    predict_pipeline = PredictPipeline()
    pred_result_nonfraud = predict_pipeline.predict(
        pred_df_nonfraud, model, preprocessor, DummyModel
    )
    print(f"Prediction result: {pred_result_nonfraud}")
    print(f"Result: {'Fraud' if pred_result_nonfraud[0] == 1 else 'Not Fraud'}")

    # Fraud example (your provided example)
    print("\n2. Testing Fraud Example:")
    data_fraud = CustomData(
        step=177,
        amount=1201681.76,
        oldbalanceOrg=1201681.76,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        type="TRANSFER",
        nameOrig="C1900756070",  # Your fraud example
        nameDest="C1995455020",  # Your fraud example
    )

    pred_df_fraud = data_fraud.get_data_as_dataframe()
    print("Input DataFrame (after tokenization):")
    print(pred_df_fraud)
    print(f"DataFrame shape: {pred_df_fraud.shape}")
    print(f"Columns: {list(pred_df_fraud.columns)}")

    pred_result_fraud = predict_pipeline.predict(
        pred_df_fraud, model, preprocessor, DummyModel
    )
    print(f"Prediction result: {pred_result_fraud}")
    print(f"Result: {'Fraud' if pred_result_fraud[0] == 1 else 'Not Fraud'}")

    # Save results
    output_dir = os.path.join("src", "pipeline", "prediction_output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prediction_results.txt"), "w") as f:
        f.write("Prediction Pipeline Test Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(
            f"Non-Fraud Example: {pred_result_nonfraud[0]} ({'Fraud' if pred_result_nonfraud[0] == 1 else 'Not Fraud'})\n"
        )
        f.write(
            f"Fraud Example: {pred_result_fraud[0]} ({'Fraud' if pred_result_fraud[0] == 1 else 'Not Fraud'})\n"
        )

    print(f"\nTest results saved to {output_dir}/prediction_results.txt")
    print("=" * 60)
