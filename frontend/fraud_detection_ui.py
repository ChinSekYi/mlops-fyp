import requests
import streamlit as st

API_URL = "http://localhost:8080"  # FastAPI backend
EXPECTED_FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Fraud Detection Demo")
st.write("Interact with the fraud detection model served by FastAPI + MLflow.")

# sample inputs
sample_nonfraud = [
    1.5345129227495835,
    0.7781897125067888,
    -0.36594124427365043,
    0.6211592355378125,
    0.4866630633753965,
    0.3216269889992719,
    0.5329776610025757,
    0.41414341646745095,
    -0.050842715582949255,
    0.28582899588507793,
    0.9301802847494782,
    -1.0264675525457934,
    0.7909155157578868,
    1.2026993067248455,
    0.6612849526709115,
    -0.450264462497494,
    0.9476729172060946,
    0.3917562652709356,
    0.2163798750841541,
    -1.5813826617669173,
    -0.3453602852370849,
    -0.20107292248689715,
    -0.39137581535132154,
    0.3797339417587787,
    -0.0725122670348631,
    -0.8224096695438422,
    -0.8863373328330746,
    -0.06410983154471686,
    -0.14832159530372568,
    -0.4230488451606895,
]

sample_fraud = [
    -1.247031078650564,
    -3.358100781230318,
    2.9033123940190624,
    -3.2122841998518754,
    1.2285399868258078,
    -3.392475595109227,
    -2.098330074491957,
    -2.3089581640614862,
    2.760195636149876,
    -1.2023172563626838,
    -1.3519447388470958,
    1.345911022966311,
    -1.0583476510963767,
    0.35117033996584174,
    -1.0496135151389292,
    0.03291560800551451,
    -1.408227854464445,
    -1.729542967577749,
    -1.643705907327757,
    0.7013402588877813,
    1.3825374453433945,
    0.48952661582995155,
    -1.2556115908772552,
    -0.6752731924825311,
    0.28136493248450095,
    2.0156212622086342,
    -0.5498479347272262,
    1.4896011560999942,
    1.0222075647733377,
    -0.012753478104972122,
]

# Sidebar navigation
menu = st.sidebar.radio(
    "Choose an action", ["Predict (Single)", "Model Info", "Metrics", "Feature Info"]
)
if menu == "Predict (Single)":
    st.subheader("Single Transaction Prediction")

    # Option to choose a sample
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
        default_values = [0.0] * len(EXPECTED_FEATURES)

    input_data = {}

    # Arrange inputs in 5 columns
    num_cols = 5
    cols = st.columns(num_cols)

    for i, feature in enumerate(EXPECTED_FEATURES):
        col = cols[i % num_cols]  # distribute evenly across 5 columns
        with col:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=float(default_values[i]),
                step=0.1,
                key=f"{feature}_input",
            )

    # Prediction button
    if st.button("Predict", key="single"):
        payload = {"features": input_data}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']} (1 = Fraud, 0 = Legit)")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

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

elif menu == "Feature Info":
    st.subheader("Feature Schema")
    if st.button("Fetch Feature Info"):
        try:
            response = requests.get(f"{API_URL}/feature-info")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")


# run streamlit webapp, do: streamlit run frontend/fraud_detection_ui.py
