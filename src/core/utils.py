"""
Utility functions for configuration loading, model evaluation, and object serialization.
"""

import os
import pickle

import pandas as pd
import yaml
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def load_environment(env_file: str = None):
    """
    Load environment variables from a file in the /env directory at the project root.
    Default is .env in /env.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_dir = os.path.join(project_root, "env")
    env_file = env_file or ".env"
    env_path = os.path.join(env_dir, env_file)
    load_dotenv(env_path)
    print(f"[ENV] Loaded environment from {env_path}")


def load_config():
    """Load configuration from the config.yaml file in the /configs directory at the project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_root, "configs", "config.yaml")
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


def balance_classes(x_train, y_train, random_state=42):
    """
    Balance classes using SMOTE oversampling to upsample minority class.

    Args:
        x_train: Training features (DataFrame)
        y_train: Training labels (Series)
        random_state: Random state for reproducibility

    Returns:
        Balanced x_train, y_train
    """
    # print(f"Original distribution:\n{y_train.value_counts()}")
    # print(f"Original fraud rate: {y_train.mean():.4f}")

    # Use SMOTE to oversample minority class (fraud cases)
    smote = SMOTE(random_state=random_state, sampling_strategy="auto")
    X_resampled, y_resampled = smote.fit_resample(x_train, y_train)

    # Convert back to pandas for consistency
    X_resampled = pd.DataFrame(X_resampled, columns=x_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    # print(f"Balanced distribution:\n{y_resampled.value_counts()}")
    # print(f"Balanced fraud rate: {y_resampled.mean():.4f}")

    return X_resampled, y_resampled


def balance_classes_smotenc(x_train, y_train, categorical_features, random_state=123):
    """
    Balance classes using SMOTENC for mixed data types (numerical + categorical).
    c
    Args:
        x_train: Training features (DataFrame)
        y_train: Training labels (Series)
        categorical_features: List of column indices that are categorical
        random_state: Random state for reproducibility

    Returns:
        Balanced x_train, y_train
    """
    # print(f"Original distribution:\n{y_train.value_counts()}")
    # print(f"Original fraud rate: {y_train.mean():.4f}")

    try:
        # Apply SMOTENC for mixed data types
        smotenc = SMOTENC(
            categorical_features=categorical_features,
            random_state=random_state,
            sampling_strategy="auto",  # Balance to 50:50
        )

        X_balanced, y_balanced = smotenc.fit_resample(x_train, y_train)

        # Convert back to pandas for consistency
        X_balanced = pd.DataFrame(X_balanced, columns=x_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)

        # print(f"Balanced distribution:\n{y_balanced.value_counts()}")
        # print(f"Balanced fraud rate: {y_balanced.mean():.4f}")

        return X_balanced, y_balanced

    except Exception as e:
        print(f"SMOTENC failed: {e}")
        print("Falling back to SMOTE...")
        return balance_classes(x_train, y_train, random_state)


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
