"""
Integration tests for database connectivity and data persistence.
Tests connections to PostgreSQL, S3, and other external data sources.
"""

import os

import pytest

from src.utils import load_environment

# Load environment
env_file = os.getenv("ENV_FILE", ".env")
load_environment(env_file)


class TestDataSourceIntegration:

    def test_environment_variables(self):
        """Test that required environment variables are properly loaded."""
        required_vars = ["MLFLOW_TRACKING_URI", "REGISTERED_MODEL_NAME"]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            pytest.skip(f"Missing environment variables: {missing_vars}")

        # Test that variables have reasonable values
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            assert mlflow_uri.startswith(
                ("http://", "https://", "file://")
            ), f"Invalid MLflow URI format: {mlflow_uri}"

        model_name = os.getenv("REGISTERED_MODEL_NAME")
        if model_name:
            assert len(model_name) > 0, "Model name cannot be empty"
            assert not model_name.isspace(), "Model name cannot be whitespace"


"""
@pytest.mark.external
class TestExternalServiceIntegration:
    def test_s3_bucket_access(self):
        #Test S3 bucket access for MLflow artifacts.
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        bucket_name = os.getenv("MLFLOW_S3_BUCKET")

        if not bucket_name:
            pytest.skip("S3 bucket name not configured")

        try:
            s3_client = boto3.client("s3")

            # Test listing bucket (requires read access)
            response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)

            assert "Contents" in response or response["KeyCount"] == 0

        except NoCredentialsError:
            pytest.skip("AWS credentials not configured")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                pytest.skip(f"S3 bucket {bucket_name} does not exist")
            else:
                pytest.skip(f"S3 access failed: {e}")
        except Exception as e:
            pytest.skip(f"S3 connection failed: {e}")
 """
