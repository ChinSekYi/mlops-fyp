# Minimal train pipeline for credit card fraud detection
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
	# Ingest data
	ingestion = DataIngestion()
	train_path, test_path = ingestion.initiate_data_ingestion()

	# Transform data
	transformer = DataTransformation()
	train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

	# Train model
	trainer = ModelTrainer()
	pred, f1 = trainer.initiate_model_trainer(train_arr, test_arr)
	print(f"Test F1 score: {f1}")
