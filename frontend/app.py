import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import requests
import streamlit as st

from frontend.utils import load_environment

# Get API URL from Docker environment variable first, then fallback to .env
"""
API_URL = os.getenv("MODEL_SERVER_IP")
if not API_URL:
    # Fallback: load from .env file
    env_file = os.getenv("ENV_FILE", ".env")
    load_environment(env_file)
    API_URL = os.getenv("MODEL_SERVER_IP")
"""
env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)

EXPECTED_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",
    "nameOrig",  # Raw string input like "C1900756070"
    "nameDest",  # Raw string input like "C1995455020"
]

TYPE_DROPDOWN_VALUES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Fraud Detection Demo")
st.write("Interact with the fraud detection model served by FastAPI + MLflow.")


# Sidebar navigation for multipage UI
page = st.sidebar.radio(
    "Select Page", ["Predict", "Model Info", "Metrics", "Feature Info"]
)

if page == "Predict":
    st.subheader("Single Transaction Prediction")

    # Sample input options
    sample_nonfraud = {
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
    sample_fraud = {
        "step": 177,
        "amount": 1201681.76,
        "oldbalanceOrg": 1201681.76,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "type": "TRANSFER",
        "nameOrig": "C1900756070",
        "nameDest": "C1995455020",
    }

    sample_choice = st.radio(
        "Load sample input?",
        ["None", "Non-Fraud Example", "Fraud Example"],
        horizontal=True,
    )
    if sample_choice == "Non-Fraud Example":
        default_values = sample_nonfraud
    elif sample_choice == "Fraud Example":
        default_values = sample_fraud
    else:
        default_values = {}
        for k in EXPECTED_FEATURES:
            if k == "type":
                default_values[k] = TYPE_DROPDOWN_VALUES[0]
            elif k in ["nameOrig", "nameDest"]:
                default_values[k] = "C0000000000"
            elif k == "step":
                default_values[k] = 1
            else:
                default_values[k] = 0.0

    input_data = {}
    num_cols = 3
    cols = st.columns(num_cols)
    for i, feature in enumerate(EXPECTED_FEATURES):
        col = cols[i % num_cols]
        with col:
            if feature == "type":
                input_data[feature] = st.selectbox(
                    "Transaction type",
                    TYPE_DROPDOWN_VALUES,
                    index=(
                        TYPE_DROPDOWN_VALUES.index(default_values[feature])
                        if default_values[feature] in TYPE_DROPDOWN_VALUES
                        else 0
                    ),
                    key=f"type_{sample_choice}_{i}",  # Include sample_choice in key
                )
            else:
                label = (
                    f"${feature}"
                    if feature
                    in [
                        "amount",
                        "oldbalanceOrg",
                        "newbalanceOrig",
                        "oldbalanceDest",
                        "newbalanceDest",
                    ]
                    else feature
                )
                if feature in ["nameOrig", "nameDest"]:
                    input_data[feature] = st.text_input(
                        f"{label} (format: C1234567890)",
                        value=str(default_values[feature]),
                        key=f"text_{feature}_{sample_choice}_{i}",  # Include sample_choice in key
                    )
                elif feature == "step":
                    input_data[feature] = st.number_input(
                        f"{label} (time step - must be positive integer)",
                        min_value=1,
                        step=1,
                        format="%d",
                        value=int(default_values[feature]),
                        key=f"num_{feature}_{sample_choice}_{i}",  # Include sample_choice in key
                    )
                else:
                    input_data[feature] = st.number_input(
                        label,
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                        value=float(default_values[feature]),
                        key=f"num_{feature}_{sample_choice}_{i}",  # Include sample_choice in key
                    )
    if st.button("Predict", key="single"):
        payload = input_data
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']} (1 = Fraud, 0 = Legit)")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

elif page == "Model Info":
    st.subheader("Model Metadata")
    if st.button("Fetch Model Info"):
        try:
            response = requests.get(f"{API_URL}/model-info")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

elif page == "Metrics":
    st.subheader("Model Metrics & Params from MLflow")
    if st.button("Fetch Metrics"):
        try:
            response = requests.get(f"{API_URL}/metrics")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

elif page == "Feature Info":
    st.subheader("Feature Schema")
    if st.button("Fetch Feature Info"):
        try:
            response = requests.get(f"{API_URL}/feature-info")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")
