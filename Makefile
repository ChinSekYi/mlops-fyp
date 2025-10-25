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
	pylint src backend frontend tests --fail-under=7

format:
	isort .
	black .

flake:
	flake8 src backend frontend tests

check: format lint flake

app: streamlit run frontend/app.py

# =========================
# MLflow Server EC2 Commands (run only on MLflow server EC2)
# =========================
# Note: requires setup -> aws configure --profile <env>-bkt
# Can only run on dedicated MLflow-server EC2
mlflow: # replace <env> with dev, staging or prod
	docker compose -f docker-compose.mlflow-server-<env>.yml up

# =========================
# App/Frontend EC2 Commands (run only on app/frontend EC2)
# =========================
# Requires `export AWS_PROFILE=<env>-bkt, export AWS_DEFAULT_PROFILE=<env>-bkt`
up-dev:
	docker compose -f infra/compose-files/docker-compose.dev.yml up

up-staging:
	docker compose -f infra/compose-files/docker-compose.staging.yml up

up-prod:
	docker compose -f infra/compose-files/docker-compose.prod.yml up

# =========================
# Model Promotion
# =========================

download-champion-dev:
	AWS_PROFILE=dev-bkt AWS_DEFAULT_PROFILE=dev-bkt ENV_FILE=env/.env.dev_machine python3 model-promotion/download_champion_model_dev.py

upload-champion-to-staging:
	AWS_PROFILE=stag-bkt AWS_DEFAULT_PROFILE=stag-bkt ENV_FILE=env/.env.stag_machine python3 model-promotion/upload_champion_model_to_staging.py

promote-model-dev:
	$(MAKE) download-champion-dev
	$(MAKE) upload-champion-to-staging
	AWS_PROFILE=dev-bkt AWS_DEFAULT_PROFILE=dev-bkt

download-champion-staging:
	AWS_PROFILE=stag-bkt AWS_DEFAULT_PROFILE=stag-bkt ENV_FILE=env/.env.stag_machine python3 model-promotion/download_champion_model_staging.py

upload-champion-to-prod:
	AWS_PROFILE=prod-bkt AWS_DEFAULT_PROFILE=prod-bkt ENV_FILE=env/.env.prod_machine python3 model-promotion/upload_champion_model_to_prod.py

promote-model-staging:
	$(MAKE) download-champion-staging
	$(MAKE) upload-champion-to-prod
	AWS_PROFILE=stag-bkt AWS_DEFAULT_PROFILE=stag-bkt

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

test-dev: # works in local
	pytest tests/unit -v

test-pipeline-ci: # works if aws profile is set to dev_bkt
	pytest tests/unit tests/integration -v --maxfail=1

test-staging: #requires the right aws bucket credentials #ensure mlflow server & banckend+frontend server are running
	pytest tests/e2e  -v

test-prod:
	pytest tests/smoke -v

# ==========================
# Others
# ==========================
aws-list-profiles:
	aws configure list-profiles
rmi:
	docker rmi $(docker images -q)
rm:
	docker container prune -f

