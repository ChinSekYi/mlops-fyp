import os
import pytest
import requests
from dotenv import load_dotenv

load_dotenv()
MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")


def test_api_prediction(sample_input):
    """Test API returns 200 and valid JSON response."""
    try:
        res = requests.post(MODEL_SERVER_IP, json=sample_input)
    except Exception as e:
        pytest.fail(f"API request failed: {e}")

    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_api_invalid_input(model_server_ip):
    """API should return 4xx for malformed input."""
    bad_input = {"features": {"Time": "wrong_type"}}
    try:
        res = requests.post(model_server_ip, json=bad_input)
    except Exception as e:
        pytest.fail(f"API request failed: {e}")

    assert 400 <= res.status_code < 500
