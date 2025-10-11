.PHONY: install download-data run-pipeline app lint test format flake check mlflow-server all

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# Data
download-data:
	dvc pull

# Pipeline/ML
run-pipeline: 
	python3 -m src.pipeline.run_pipeline

# App
app:
	streamlit run frontend/fraud_detection_ui.py

api-server:
	uvicorn api.main:app --host 0.0.0.0 --port 5050 --reload

# Docker
mlflow-server: 
	docker compose -f docker/docker-compose.yml up

# Linting/Quality
lint:
	pylint src api tests --fail-under=7

format:
	isort .
	black .

flake:
	flake8 src api tests

check: format lint flake

# Test
test:
	pytest -v --disable-warnings tests/


# Testing Strategies for Different Environments
test-dev:
	pytest tests/unit -v

test-pipeline-ci:
	pytest tests/unit tests/integration -v --maxfail=1

test-staging:
	pytest tests/integration tests/e2e tests/smoke -v

test-prod:
	pytest tests/smoke -v

all: install format test lint download-data mlflow-server run-pipeline app