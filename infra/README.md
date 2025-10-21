# Infra

This folder contains docker compose files and docker files.

## Usage
- Use the appropriate Docker Compose file to spin up services for your target environment.
- See the Makefile for commands to spin up services.

## Key Makefile Commands

There are two types of EC2 instances for each environment:
- **Frontend-Backend EC2:** Runs the app and backend containers
- **MLflow Server EC2:** Runs the MLflow tracking server

Use these commands on the correct EC2:

### Frontend-Backend EC2
- **Dev:**
	```sh
	make up-dev
	```
- **Staging:**
	```sh
	make up-staging
	```
- **Prod:**
	```sh
	make up-prod
	```

### MLflow Server EC2
- **Any environment:**
	```sh
	make mlflow ENV=.env.mlflow_<env>
	```
Replace `<env>` with `dev`, `staging`, or `prod` as needed.
