#!/usr/bin/env python3
"""
Test helper script to start the API server for integration testing.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_api_health(url="http://localhost:5050", timeout=30):
    """Check if API server is responding."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/", timeout=2)
            if response.status_code in [200, 405]:  # Any response means server is up
                print(f"✅ API server is responding at {url}")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    print(f"API server not responding at {url}")
    return False


def start_api_server():
    """Start the API server for testing."""
    print("Starting API server for testing...")

    # Check if server is already running
    if check_api_health(timeout=2):
        print("API server is already running")
        return True

    # Start the server
    try:
        cmd = ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5050"]
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project_root
        ) as process:
            # Wait a moment for server to start
            time.sleep(3)

            # Check if server started successfully
            if check_api_health():
                print(f"API server started successfully (PID: {process.pid})")
                return True

            print("Failed to start API server")
            process.terminate()
            return False

    except (subprocess.SubprocessError, OSError) as e:
        print(f"Error starting API server: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API server test helper")
    parser.add_argument(
        "--check", action="store_true", help="Just check if server is running"
    )
    parser.add_argument("--start", action="store_true", help="Start the server")

    args = parser.parse_args()

    if args.check:
        if check_api_health():
            sys.exit(0)
        else:
            sys.exit(1)
    elif args.start:
        if start_api_server():
            print("Server started. Press Ctrl+C to stop.")
            try:
                # Keep the script running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                sys.exit(0)
        else:
            sys.exit(1)
    else:
        parser.print_help()
