"""
Integration tests for the FastAPI prediction service.
Tests the API endpoints with real model loading and predictions.
"""

import os

import pytest
import requests
from requests.exceptions import ConnectionError

from src.utils import load_environment

# Use ENV_FILE if set, otherwise default to .env
env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)

MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP", "http://localhost:8080")

# Extract predict endpoint and base URL
if MODEL_SERVER_IP.endswith("/predict"):
    BASE_URL = MODEL_SERVER_IP.replace("/predict", "")
    PREDICT_URL = MODEL_SERVER_IP
else:
    BASE_URL = MODEL_SERVER_IP
    PREDICT_URL = f"{MODEL_SERVER_IP}/predict"

# Validate that MODEL_SERVER_IP is properly set
if not MODEL_SERVER_IP:
    MODEL_SERVER_IP = "http://localhost:8080"
    BASE_URL = MODEL_SERVER_IP
    PREDICT_URL = f"{MODEL_SERVER_IP}/predict"

print(f"Testing API base at: {BASE_URL}")
print(f"Testing predictions at: {PREDICT_URL}")


def check_api_availability():
    """Check if the API server is available."""
    try:
        # Try base URL first
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code in [
            200,
            405,
            404,
        ]  # Any response means server is up
    except requests.exceptions.RequestException:
        try:
            # Try a HEAD request to the predict endpoint
            response = requests.head(PREDICT_URL, timeout=2)
            return response.status_code in [
                200,
                405,
                404,
                422,
            ]  # 422 is validation error but server is up
        except requests.exceptions.RequestException:
            return False


class TestAPIIntegration:
    """Integration tests for the prediction API service."""

    def setup_method(self):
        """Setup method to check API availability before each test."""
        if not check_api_availability():
            pytest.skip("API server is not running or not accessible")

    def test_api_connectivity(self):
        """Test basic API server connectivity."""
        try:
            # Test if we can reach the server (any response means it's up)
            res = requests.get(f"{BASE_URL}/", timeout=5)

            # Print debug information
            print(f"Response status: {res.status_code}")
            print(f"Response headers: {res.headers}")

            # Any HTTP response means the server is running
            assert res.status_code in [200, 404, 405, 422]

        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - connection refused")
        except requests.exceptions.Timeout:
            pytest.skip("API server not responding - timeout")
        except Exception as e:
            pytest.fail(f"Connectivity test failed: {e}")

    def test_api_health_check(self):
        """Test API health check endpoint."""
        try:
            res = requests.get(f"{BASE_URL}/", timeout=5)
            assert res.status_code == 200

            data = res.json()
            assert "status" in data
            assert data["status"] == "ok"
            assert "model" in data
            assert "alias" in data

            print(
                f"Health check passed - Model: {data.get('model')}, Alias: {data.get('alias')}"
            )

        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - connection refused")
        except requests.exceptions.Timeout:
            pytest.skip("API server not responding - timeout")
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")

    def test_predict_endpoint_exists(self):
        """Test that the predict endpoint exists and accepts POST requests."""
        try:
            # Send a POST request with empty data to check if endpoint exists
            res = requests.post(PREDICT_URL, json={}, timeout=5)

            # We expect validation error (422) or some other response, not 404
            assert res.status_code != 404, "Predict endpoint not found"

            print(f"Predict endpoint status: {res.status_code}")
            print(f"Response: {res.text[:200]}")  # First 200 chars

        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - connection refused")
        except requests.exceptions.Timeout:
            pytest.skip("API server not responding - timeout")
        except Exception as e:
            pytest.fail(f"Predict endpoint test failed: {e}")

    def test_api_prediction_valid_input(self, sample_input):
        """Test API returns 200 and valid JSON response for valid input."""
        try:
            res = requests.post(PREDICT_URL, json=sample_input, timeout=10)
        except ConnectionError:
            pytest.skip("API server not running")
        except Exception as e:
            pytest.fail(f"API request failed: {e}")

        assert res.status_code == 200
        data = res.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] in [0, 1]  # Should be binary classification

    def test_api_prediction_different_transaction_types(self):
        """Test API handles different transaction types correctly."""
        transaction_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

        for trans_type in transaction_types:
            test_input = {
                "step": 1,
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "type": trans_type,
                "nameOrig_token": 123,
                "nameDest_token": 456,
            }

            try:
                res = requests.post(PREDICT_URL, json=test_input)
                assert res.status_code == 200
                data = res.json()
                assert "prediction" in data
                assert data["prediction"] in [0, 1]
            except ConnectionError:
                pytest.skip("API server not running")
            except Exception as e:
                pytest.fail(f"API request failed for {trans_type}: {e}")

    def test_api_invalid_input_missing_fields(self):
        """API should return 422 for missing required fields."""
        bad_input = {
            "step": 1,
            "amount": 181.0,
            # Missing required fields
        }

        try:
            res = requests.post(PREDICT_URL, json=bad_input)
            assert res.status_code == 422  # Validation error
        except ConnectionError:
            pytest.skip("API server not running")
        except Exception as e:
            pytest.fail(f"API request failed: {e}")

    def test_api_invalid_input_wrong_types(self):
        """API should return 422 for wrong data types."""
        bad_input = {
            "step": "not_a_number",  # Should be int
            "amount": "also_not_a_number",  # Should be float
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "type": "CASH_OUT",
            "nameOrig_token": 123,
            "nameDest_token": 456,
        }

        try:
            res = requests.post(PREDICT_URL, json=bad_input)
            assert res.status_code == 422  # Validation error
        except ConnectionError:
            pytest.skip("API server not running")
        except Exception as e:
            pytest.fail(f"API request failed: {e}")

    def test_api_invalid_transaction_type(self):
        """API should handle invalid transaction types gracefully."""
        bad_input = {
            "step": 1,
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "type": "INVALID_TYPE",  # Invalid transaction type
            "nameOrig_token": 123,
            "nameDest_token": 456,
        }

        try:
            res = requests.post(PREDICT_URL, json=bad_input)
            # Should either validate the enum or handle gracefully
            assert res.status_code in [200, 422]
        except ConnectionError:
            pytest.skip("API server not running")
        except Exception as e:
            pytest.fail(f"API request failed: {e}")

    def test_api_model_info_endpoint(self):
        """Test the model info endpoint returns model metadata."""
        try:
            res = requests.get(f"{BASE_URL}/model-info")
            assert res.status_code == 200
            data = res.json()
            # Should contain model information
            assert isinstance(data, dict)
        except ConnectionError:
            pytest.skip("API server not running")
        except Exception as e:
            pytest.fail(f"Model info request failed: {e}")
