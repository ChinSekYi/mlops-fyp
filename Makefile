.PHONY: install download-data run-pipeline app lint test format flake check mlflow-server all

# =========================
# Universal Commands (run anywhere)
# =========================
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

run-pipeline: 
	python3 -m src.pipeline.run_pipeline

lint:
	pylint src api tests --fail-under=7

format:
	isort .
	black .

flake:
	flake8 src api tests

check: format lint flake



# =========================
# MLflow Server EC2 Commands (run only on MLflow server EC2)
# =========================
# Note: requires setup -> aws configure --profile <env>-bkt
mlflow-dev-up:
	docker compose -f docker/docker-compose.mlflow-server-dev.yml up

mlflow-staging-up:
	docker compose -f docker/docker-compose.mlflow-server-staging.yml up

mlflow-prod-up:
	docker compose -f docker/docker-compose.mlflow-server-prod.yml up


# =========================
# App/Frontend EC2 Commands (run only on app/frontend EC2)
# =========================
# Requires `export AWS_PROFILE=<env>-bkt, export AWS_DEFAULT_PROFILE=<env>-bkt`
up-dev:
	docker compose -f docker/docker-compose.dev.yml up
up-staging:
	docker compose -f docker/docker-compose.staging.yml up
up-prod:
	docker compose -f docker/docker-compose.prod.yml up

# =========================
# Data Commands (run anywhere with DVC configured)
# =========================
# Requires `aws configure --profile <env>-raw`
download-data-dev:
	dvc pull -r dev

download-data-staging:
	dvc pull -r staging

download-data-prod:
	dvc pull -r prod

# =========================
# Test Commands
# =========================
test-pipeline:
	python3 -m src.components.data_ingestion
	python3 -m src.components.data_transformation
	python3 -m src.components.model_trainer
	python3 -m src.pipeline.predict_pipeline
	python3 -m src.pipeline.run_pipeline

test:
	pytest -v --disable-warnings tests/

test-dev:
	pytest tests/unit -v

test-pipeline-ci:
	pytest tests/unit tests/integration -v --maxfail=1

test-staging:
	pytest tests/integration tests/e2e  -v

test-prod:
	pytest tests/smoke -v

# =========================
# Others
# =========================
aws-list-profiles:
	aws configure list-profiles
rmi:
	docker rmi $(docker images -q)