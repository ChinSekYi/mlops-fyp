"""
Utility functions for configuration loading, model evaluation, and object serialization.
"""

import os
import pickle

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


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


def balance_classes(x_train, y_train, random_state=123):
    """Upsample the minority class in x_train/y_train to match the majority class size."""
    train_balanced = x_train.copy()
    train_balanced["isFraud"] = y_train.values
    majority_class = train_balanced["isFraud"].value_counts().idxmax()
    minority_class = train_balanced["isFraud"].value_counts().idxmin()
    df_majority = train_balanced[train_balanced["isFraud"] == majority_class]
    df_minority = train_balanced[train_balanced["isFraud"] == minority_class]
    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len(df_majority), random_state=random_state
    )
    train_balanced = pd.concat([df_majority, df_minority_upsampled])
    train_balanced = train_balanced.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    x_train_bal = train_balanced.drop("isFraud", axis=1)
    y_train_bal = train_balanced["isFraud"]
    return x_train_bal, y_train_bal


def one_hot_encode_and_align(x_train, x_test, column):
    # One-hot encode the specified column in both train and test
    for df in [x_train, x_test]:
        if column in df.columns:
            dummies = pd.get_dummies(df[column], prefix=f"{column}_")
            df.drop([column], axis=1, inplace=True)
            df[dummies.columns] = dummies
    # Align columns to ensure both have the same features
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)
    return x_train, x_test


def standardize_columns(x_train, x_test, col_names):
    scaler = StandardScaler().fit(x_train[col_names].values)
    x_train[col_names] = scaler.transform(x_train[col_names].values)
    x_test[col_names] = scaler.transform(x_test[col_names].values)
    return x_train, x_test


def tokenize_column(x_train, x_test, column):
    # Fit factorizer on x_train
    x_train[f"{column}_token"], uniques = pd.factorize(x_train[column])
    # Map test set using train uniques
    x_test[f"{column}_token"] = uniques.get_indexer(x_test[column])
    # Replace -1 (unknowns) with a new integer
    x_test[f"{column}_token"] = x_test[f"{column}_token"].replace(-1, len(uniques))
    return x_train, x_test


# not used
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


if __name__ == "__main__":
    load_config()
