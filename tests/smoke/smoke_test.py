import os

import requests


def test_smoke():
    API_URL = os.getenv("MODEL_SERVER_IP", "http://localhost:8000/predict")
    sample_input = {
        "step": 1,
        "amount": 100.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 900.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "type": "CASH_OUT",
        "nameOrig": "C123456789",
        "nameDest": "C987654321",
    }
    res = requests.post(API_URL, json=sample_input, timeout=5)
    assert res.status_code == 200, f"Status code: {res.status_code}"
    data = res.json()
    assert "prediction" in data, "Missing prediction in response"
