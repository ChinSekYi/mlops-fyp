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

# Local API server (for development without Docker)
api-server-local:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
mlflow-server: 
	docker compose -f docker/docker-compose.yml up

# Staging on same EC2 (different ports)
mlflow-server-staging:
	docker compose -f docker/docker-compose.staging.yml up

mlflow-server-prod:
	docker compose -f docker/docker-compose.prod.yml up


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
	pytest tests/integration tests/e2e  -v

test-prod:
	pytest tests/smoke -v


rmi:
	docker rmi $(docker images -q)