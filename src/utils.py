"""
Utility functions for configuration loading, model evaluation, and object serialization.
"""

import os
import pickle
from dotenv import load_dotenv
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_environment(env_file: str = None):
    """
    Load environment variables from a file.
    Default is .env at project root.
    """
    if env_file is None:
        env_file = ".env"

    load_dotenv(env_file)
    print(f"[ENV] Loaded environment from {env_file}")

    # Optional: ensure ENV is set
    os.environ.setdefault("ENV", "dev")

def load_config():
    """Load configuration from the config.yaml file at the project root."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    Args:
        file_path (str): Path to save the object.
        obj: Python object to serialize.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def evaluate_model(x, y):
    """
    Evaluate a model's predictions using common classification metrics.
    Args:
        x: Predicted labels.
        y: True labels.
    Returns:
        dict: Dictionary with accuracy, precision, recall, and f1 scores.
    """
    report = {
        "accuracy": accuracy_score(x, y),
        "precision": precision_score(x, y),
        "recall": recall_score(x, y),
        "f1": f1_score(x, y),
    }
    return report


def load_object(file_path):
    """
    Load an object from a file using pickle.
    Args:
        file_path (str): Path to the pickle file.
    Returns:
        The loaded Python object.
    """
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)
