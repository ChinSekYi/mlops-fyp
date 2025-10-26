# DSA4288 Honours Year Project: MLOps for Traditional ML
Using Fraud Detection as a use case.

[![Docker Image CI](https://github.com/ChinSekYi/mlops-fyp/actions/workflows/ci.yml/badge.svg)](https://github.com/ChinSekYi/mlops-fyp/actions/workflows/ci.yml)

A full-stack MLOps pipeline for transaction fraud detection using MLflow, FastAPI, Streamlit, DVC, AWS, and Docker.

---

## Architecture Diagram

<div align="center">
    <img src="docs/images/doc_mlops_blueprint.jpg" alt="MLOps End-to-End Blueprint"/>
    <br>
    <span>MLOps End-to-End Blueprint</span>
</div>

---

## File Structure

```
mlops-fyp/
├── README.md
├── Makefile
├── requirements.txt
├── artifacts/
│   ├── metrics/
│   ├── models/
│   ├── preprocessor/
├── backend/
│   ├── main.py
│   ├── utils.py
├── configs/
├── data/
│   ├── processed/
│   ├── raw/
├── docs/
│   ├── Developer_Guide/
│   ├── User_Guide/
│   ├── images/
├── env/
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   ├── utils.py
├── infra/
│   ├── compose-files/
│   ├── docker/
├── model-promotion/
├── notebooks/
├── requirements/
├── src/
│   ├── components/
│   ├── core/
│   ├── pipeline/
├── tests/
```

---

## Features
- Experiment tracking and model registry with MLflow
- REST API for model inference (FastAPI)
- Interactive frontend (Streamlit)
- Data versioning (DVC) and S3 storage
- Automated CI/CD with GitHub Actions and Docker

## Quickstart
1. Clone the repo and set up your environment (see `/docs/Developer_Guide/backend-frontend-machine-setup.md` and `/docs/Developer_Guide/data-management.md`).
2. Build and manage Docker images (see `/docs/Developer_Guide/docker-setup.md`).
3. Set up MLflow tracking server (see `/docs/Developer_Guide/mlflow-tracking-server-setup.md`).
4. Set up SSH for GitHub (see `/docs/Developer_Guide/git-ssh-setup.md`).
5. Reference CI/CD secrets (see `/docs/Developer_Guide/github-actions-secrets.md`).
6. Run the pipeline and serve models (see `/docs/User_Guide/model-experimentation-guide.md` and `/docs/User_Guide/model-serving-guide.md`).
7. For detailed setup, troubleshooting, and deployment, see the [Documentation folder](./docs/README.md).

## Recommended Reading Order
1. Developer Guide (setup, environment, Docker, MLflow, SSH, secrets)
2. User Guide (experimentation, serving)
3. Images (architecture, UI screenshots)

## Data Source
- [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)