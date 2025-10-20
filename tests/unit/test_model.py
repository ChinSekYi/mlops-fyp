"""
Unit tests for the Model
Tests model load, shape
"""

import warnings

import numpy as np
import pandas as pd

from src.core.utils import one_hot_encode_and_align
from src.pipeline.predict_pipeline import CustomData

warnings.filterwarnings("ignore")


def test_model_load(model):
    """Check that the model can be loaded."""
    assert model is not None


def test_model_prediction_shape(model, preprocessor):
    """Check that model returns correct output length on dummy input."""
    # Always preprocess input to match model expectation
    sample_input = CustomData(
        step=1,
        amount=1000.0,
        oldbalanceOrg=5000.0,
        newbalanceOrig=4000.0,
        oldbalanceDest=0.0,
        newbalanceDest=1000.0,
        type="CASH_OUT",
        nameOrig="C84071102",
        nameDest="C1576697216",
    ).get_data_as_dataframe()

    # One-hot encode 'type' to match model features
    # Use a dummy train DataFrame with all possible types for alignment
    all_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    dummy_train = sample_input.copy()
    dummy_train = pd.concat(
        [
            CustomData(
                step=1,
                amount=1000.0,
                oldbalanceOrg=5000.0,
                newbalanceOrig=4000.0,
                oldbalanceDest=0.0,
                newbalanceDest=1000.0,
                type=t,
                nameOrig="C84071102",
                nameDest="C1576697216",
            ).get_data_as_dataframe()
            for t in all_types
        ],
        ignore_index=True,
    )

    dummy_train_encoded, sample_input_encoded = one_hot_encode_and_align(
        dummy_train, sample_input, "type"
    )

    if preprocessor is not None:
        data_scaled = preprocessor.transform(sample_input_encoded)
        preds = model.predict(data_scaled)
    else:
        preds = model.predict(sample_input_encoded)

    print(f"preds is {preds}")
    assert len(preds) == 1
    assert isinstance(preds[0], (int, float, np.integer, np.floating))
