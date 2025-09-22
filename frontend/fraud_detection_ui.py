import streamlit as st
import requests

API_URL = "http://localhost:8080"  # FastAPI backend

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Fraud Detection Demo")
st.write("Interact with the fraud detection model served by FastAPI + MLflow.")

# Sidebar navigation
menu = st.sidebar.radio(
    "Choose an action",
    ["Predict (Single)", "Predict (Batch)", "Model Info", "Metrics", "Feature Info"]
)

if menu == "Predict (Single)":
    st.subheader("Single Transaction Prediction")
    features = [st.number_input(f"Feature {i+1}", value=0.0, step=0.1) for i in range(30)]

    if st.button("Predict", key="single"):
        payload = {"features": features}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

elif menu == "Predict (Batch)":
    st.subheader("Batch Transaction Prediction")
    st.write("Enter multiple rows of features (comma-separated). Example: `0.1,0.2,...,0.3` (30 values).")

    batch_input = st.text_area("Paste multiple rows (one per line):", height=200)

    if st.button("Batch Predict"):
        try:
            rows = [list(map(float, line.split(","))) for line in batch_input.strip().split("\n") if line.strip()]
            payload = {"features_list": rows}
            response = requests.post(f"{API_URL}/batch-predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.json(result)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Invalid input: {e}")

elif menu == "Model Info":
    st.subheader("Model Metadata")
    if st.button("Fetch Model Info"):
        try:
            response = requests.get(f"{API_URL}/model-info")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

elif menu == "Metrics":
    st.subheader("Model Metrics & Params from MLflow")
    if st.button("Fetch Metrics"):
        try:
            response = requests.get(f"{API_URL}/metrics")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

# 6️⃣ Feature Info
elif menu == "Feature Info":
    st.subheader("Feature Schema")
    if st.button("Fetch Feature Info"):
        try:
            response = requests.get(f"{API_URL}/feature-info")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")


# run streamlit webapp, do: streamlit run frontend/fraud_detection_ui.py
