.PHONY: install download-data run-pipeline app lint test format mlflow-server all

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
	pylint src/ api/ tests/

test:
	pytest -v tests/

format:
	isort *.py
	black *.py

# Docker
mlflow-server: 
	docker compose -f docker/docker-compose.yml up

all: install format test lint download-data mlflow-server run-pipeline app