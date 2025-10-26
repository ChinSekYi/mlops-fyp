# Docker Developer Guide

*This guide explains how to build, push, and pull Docker images for backend, frontend, and MLflow services.*

## When to Build & Push Images
- After making changes to backend, frontend, or MLflow code.
- When updating dependencies or Dockerfiles.
- Before deploying to EC2 or other cloud servers.

## Image Naming & Tagging
- Use your Docker Hub username (e.g., `sychin0606`).
- Tag images with `:latest` for general use, or use semantic tags (e.g., `:v1.2.0`) for releases.

## Build and Push Images to Docker Hub
1. **Log in to Docker Hub:**
    ```sh
    docker login
    ```
    Enter your Docker Hub username and password when prompted.

2. **Build images:**
    ```
    docker build -t sychin0606/mlflow-server:latest -f docker/Dockerfile.mlflow .
    docker build -t sychin0606/backend:latest -f docker/Dockerfile.backend .
    docker build -t sychin0606/frontend:latest -f docker/Dockerfile.frontend .
    ```

3. **Push images:**
    ```sh
    docker push sychin0606/mlflow-server:latest
    docker push sychin0606/backend:latest
    docker push sychin0606/frontend:latest
    ```

## Using Docker Compose
- If your `docker-compose.yml` specifies images (e.g., `image: sychin0606/frontend:latest`), you can build all images:
    ```sh
    docker compose build
    ```
- Then push each image as above.

## Pulling Images (on EC2 or local machine)
- To update containers with the latest images:
    ```sh
    docker pull sychin0606/mlflow-server:latest
    docker pull sychin0606/backend:latest
    docker pull sychin0606/frontend:latest
    ```
- Or with Docker Compose:
    ```sh
    docker compose -f docker/docker-compose.yml pull
    docker compose -f docker/docker-compose.yml up -d
    ```