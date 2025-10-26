# GitHub Actions Secrets Reference

*This file documents the secrets used in GitHub Actions workflows for CI/CD and deployment.*

---

## Secrets Used in CI Workflow (`ci.yml`)

| Secret Name           | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `AWS_ACCESS_KEY_ID`  | AWS access key for authentication (used for access to s3 bucket)         |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for authentication (used for access to s3 bucket)        |
| `AWS_REGION`         | AWS region for resources (e.g., `ap-southeast-1`)                  |
| `AWS_S3_BUCKET_CI`   | S3 bucket name for CI artifacts or data storage                    |

---

## Secrets Used in Staging Deployment Workflow (`cd-staging.yml`)

| Secret Name                | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `DOCKERHUB_USERNAME`      | Docker Hub username for pushing/pulling images                               |
| `DOCKERHUB_TOKEN`         | Docker Hub access token for authentication                                   |
| `MLOPS_STAGING_MACHINE_NAME` | Hostname for the staging machine                            |
| `EC2_SSH_KEY`             | SSH private key for connecting to EC2 |
| `ENV_STAGING_CONTENTS`    | Contents of the staging environment file (`.env.staging_machine`) for deployment |

---

## Secrets Used in Production Deployment Workflow (`ci-prod.yml`)

| Secret Name                | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `DOCKERHUB_USERNAME`      | Docker Hub username for pushing/pulling images                               |
| `DOCKERHUB_TOKEN`         | Docker Hub access token for authentication                                   |
| `MLOPS_PROD_MACHINE_NAME` | Hostname for the production machine                            |
| `EC2_SSH_KEY`             | SSH private key for connecting to EC2|
| `ENV_PROD_CONTENTS`       | Contents of the production environment file (`.env.prod_machine`) for deployment |
