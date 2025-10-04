import requests
import streamlit as st

API_URL = "http://localhost:8080"
EXPECTED_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",
    "nameOrig_token",
    "nameDest_token"
]

TYPE_DROPDOWN_VALUES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Fraud Detection Demo")
st.write("Interact with the fraud detection model served by FastAPI + MLflow.")

st.subheader("Single Transaction Prediction")

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
                key=f"type_{i}"
            )
        else:
            label = f"${feature}" if feature in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"] else feature
            input_data[feature] = st.number_input(
                label,
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key=f"num_{feature}_{i}"
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
