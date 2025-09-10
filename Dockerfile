FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY src /src


# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose MLflow port
EXPOSE 5050

# Run MLflow using environment variables
CMD ["sh", "-c", "mlflow server \
  -h 0.0.0.0 \
  -p 5050 \
  --backend-store-uri postgresql://$DB_USER:$DB_PASSWORD@$REAL_DB_ENDPOINT:$DB_PORT/$DB_NAME \
  --default-artifact-root s3://$MLFLOW_S3_BUCKET"]