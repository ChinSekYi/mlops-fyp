# Model Promotion Scripts

This folder contains scripts for promoting MLflow models between environments (dev, staging, prod) across multiple EC2 instances. Each environment has two EC2s: one for the MLflow server and one for the backend/app. Model promotion is the process of transferring a registered model (with a specific alias, e.g., `champion`) from one stage to the next.

## Scripts Overview

- **download_champion_model_dev.py**
  - Downloads the current `champion` model from the dev MLflow server.
  - Use when you want to promote the latest dev model to staging.
  - Run via Makefile: `make download-champion-dev`

- **upload_champion_model_to_staging.py**
  - Uploads the downloaded dev `champion` model to the staging MLflow server and registers it.
  - Use after running the dev download script, to promote the model to staging.
  - Run via Makefile: `make upload-champion-to-staging`

- **download_champion_model_staging.py**
  - Downloads the current `champion` model from the staging MLflow server.
  - Use when you want to promote the latest staging model to prod.
  - Run via Makefile: `make download-champion-staging`

- **upload_champion_model_to_prod.py**
  - Uploads the downloaded staging `champion` model to the prod MLflow server and registers it.
  - Use after running the staging download script, to promote the model to prod.
  - Run via Makefile: `make upload-champion-to-prod`

## Promotion Workflow

- **Dev → Staging**: Run `make promote-model-dev` (downloads from dev, uploads to staging)
- **Staging → Prod**: Run `make promote-model-staging` (downloads from staging, uploads to prod)

## When to Use
- Use these scripts when you want to promote a validated model from one environment to the next (e.g., after successful testing in dev or staging).
- Always ensure you are using the correct AWS profile and environment file for each stage.

See the Makefile for exact commands and environment requirements.
