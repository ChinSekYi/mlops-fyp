"""
Unit tests for the Model
Tests model load, shape
"""

import numpy as np
import pandas as pd

from src.core.utils import tokenize_column


def test_model_load(model):
    """Check that the model can be loaded."""
    assert model is not None


def test_model_prediction_shape(model, preprocessor):
    """Check that model returns correct output length on dummy input."""
    # PaySim dataset in RAW format (before preprocessing):
    sample_input = pd.DataFrame(
        {
            "step": [1],
            "amount": [1000.0],
            "oldbalanceOrg": [5000.0],
            "newbalanceOrig": [4000.0],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [1000.0],
            "type": ["CASH_OUT"],
            "nameOrig": ["C84071102"],
            "nameDest": ["C1576697216"],
        }
    )

    if preprocessor is not None:
        # Apply tokenization
        sample_copy = sample_input.copy()
        sample_dummy = sample_input.copy()
        sample_tokenized, _ = tokenize_column(sample_copy, sample_dummy, "nameOrig")
        sample_tokenized, _ = tokenize_column(
            sample_tokenized, sample_dummy, "nameDest"
        )
        sample_tokenized = sample_tokenized.drop(["nameOrig", "nameDest"], axis=1)

        # Apply preprocessor
        data_scaled = preprocessor.transform(sample_tokenized)
        preds = model.predict(data_scaled)
    else:
        # No preprocessor, model should handle raw data directly
        preds = model.predict(sample_input)

    print(f"preds is {preds}")
    assert len(preds) == 1
    assert isinstance(preds[0], (int, float, np.integer, np.floating))
