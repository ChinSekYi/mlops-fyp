import os

import requests
from dotenv import load_dotenv

load_dotenv()
MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP", "http://localhost:8000")
PREDICT_URL = f"{MODEL_SERVER_IP}/predict"


def test_end_to_end(sample_input):
    """Simulate UI -> API -> Model -> UI flow."""
    # Send request to API
    res = requests.post(PREDICT_URL, json=sample_input)
    assert res.status_code == 200
    data = res.json()

    # Simulate minimal UI handling
    prediction = data.get("prediction")
    assert prediction is not None
    assert isinstance(prediction, (int, float))
