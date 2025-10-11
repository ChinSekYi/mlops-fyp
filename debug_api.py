#!/usr/bin/env python3
"""
Debug script to test the API endpoints and see what's happening.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP", "http://localhost:8080")
PREDICT_URL = f"{MODEL_SERVER_IP}/predict"

print(f"MODEL_SERVER_IP: {MODEL_SERVER_IP}")
print(f"PREDICT_URL: {PREDICT_URL}")

# Test sample input
sample_input = {
    'amount': 181.0,
    'nameDest_token': 456,
    'nameOrig_token': 123,
    'newbalanceDest': 0.0,
    'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0,
    'oldbalanceOrg': 181.0,
    'step': 1,
    'type__CASH_IN': 0,
    'type__CASH_OUT': 1,
    'type__DEBIT': 0,
    'type__PAYMENT': 0,
    'type__TRANSFER': 0
}

print("\n1. Testing health check endpoint...")
try:
    health_response = requests.get(f"{MODEL_SERVER_IP}/", timeout=5)
    print(f"Health check status: {health_response.status_code}")
    print(f"Health check response: {health_response.text}")
except Exception as e:
    print(f"Health check failed: {e}")

print("\n2. Testing predict endpoint...")
try:
    predict_response = requests.post(PREDICT_URL, json=sample_input, timeout=10)
    print(f"Predict status: {predict_response.status_code}")
    print(f"Predict response: {predict_response.text}")
    
    if predict_response.status_code != 200:
        print(f"Response headers: {dict(predict_response.headers)}")
        
except Exception as e:
    print(f"Predict request failed: {e}")

print("\n3. Testing if server responds to any endpoint...")
try:
    # Try some other endpoints
    endpoints_to_test = ["/", "/docs", "/model-info", "/metrics", "/feature-info"]
    
    for endpoint in endpoints_to_test:
        try:
            url = f"{MODEL_SERVER_IP}{endpoint}"
            response = requests.get(url, timeout=5)
            print(f"{endpoint}: {response.status_code}")
        except:
            print(f"{endpoint}: Failed")
            
except Exception as e:
    print(f"Endpoint testing failed: {e}")
