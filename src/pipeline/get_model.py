import mlflow
import os

model_id = 'm-5793c91765c748d989118aa7e93e6bd8'
logged_model = mlflow.get_logged_model(model_id)
print(logged_model, logged_model.metrics, logged_model.params)


experimend_id = '254433368725384354'

# Searching By Metrics
# Find high-performing models
high_accuracy_models = mlflow.search_logged_models(
    experiment_ids=[experimend_id],  # Replace with your experiment ID
    filter_string="metrics.test_accuracy > 0.9",
)

print(high_accuracy_models)