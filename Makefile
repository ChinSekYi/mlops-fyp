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

# Linting/Testing/Quality
lint:
	pylint src api tests --fail-under=7

format:
	isort .
	black .

flake:
	flake8 src api tests

check: format lint flake

test:
	pytest -v tests/

# Docker
mlflow-server: 
	docker compose -f docker/docker-compose.yml up

all: install format test lint download-data mlflow-server run-pipeline app