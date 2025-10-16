# pylint: skip-file
# flake8: noqa

import os

import pandas as pd

from src.exception import CustomException
from src.utils import load_config, load_object, tokenize_column

config = load_config()

predict_config = config["predict_pipeline"]
model_path = predict_config["model_path"]
preprocessor_path = predict_config["preprocessor_path"]


class PredictPipeline:
    def predict(self, features):
        try:
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred_result = model.predict(data_scaled)
            return pred_result
        except Exception as e:
            import sys

            raise CustomException(e, sys)


class CustomData:
    """
    Responsible for taking PaySim transaction inputs (RAW FORMAT with string account names)
    and converting them to the format expected by the fraud detection model.
    """

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
                "nameOrig": [self.nameOrig],  # Raw account ID
                "nameDest": [self.nameDest],  # Raw account ID
            }
            df = pd.DataFrame(custom_data_input)

            # Step 2: Tokenize account names (same approach as API)
            try:
                # Use the same tokenization approach as the API with dummy dataframe
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
            # Preprocessor expects: numerical features + categorical features
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
    pred_result_nonfraud = predict_pipeline.predict(pred_df_nonfraud)
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

    pred_result_fraud = predict_pipeline.predict(pred_df_fraud)
    print(f"Prediction result: {pred_result_fraud}")
    print(f"Result: {'Fraud' if pred_result_fraud[0] == 1 else 'Not Fraud'}")

    # Save results
    output_dir = os.path.join("src", "prediction_output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prediction_test.txt"), "w") as f:
        f.write("Prediction Pipeline Test Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(
            f"Non-Fraud Example: {pred_result_nonfraud[0]} ({'Fraud' if pred_result_nonfraud[0] == 1 else 'Not Fraud'})\n"
        )
        f.write(
            f"Fraud Example: {pred_result_fraud[0]} ({'Fraud' if pred_result_fraud[0] == 1 else 'Not Fraud'})\n"
        )

    print(f"\nTest results saved to {output_dir}/prediction_test.txt")
    print("=" * 60)
