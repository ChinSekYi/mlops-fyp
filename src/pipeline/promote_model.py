from mlflow import MlflowClient

client = MlflowClient()

# Names of models in each environment
src_model = "dev.ml_team.bankruptcy_model"
dst_model = "staging.ml_team.bankruptcy_model"

# Version to promote
src_version = 5

# Copy model version from dev to staging
client.copy_model_version(
    src_model_uri=f"models:/{src_model}/{src_version}", dst_name=dst_model
)

print(f"✅ Model version {src_version} promoted from {src_model} to {dst_model}")
