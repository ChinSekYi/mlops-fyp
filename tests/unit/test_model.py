"""
Unit tests for the Model
Tests model load, shape
"""

import warnings

import numpy as np
import pytest

from src.pipeline.predict_pipeline import CustomData

warnings.filterwarnings("ignore")


def test_model_load(model):
    """Check that the model can be loaded."""
    assert model is not None


def test_model_prediction_shape(model, preprocessor):
    """Check that model returns correct output length on dummy input."""
    # Prepare raw input matching model expectation
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

    if preprocessor is None:
        pytest.skip("Preprocessor is required for model prediction shape test.")
    data_scaled = preprocessor.transform(sample_input)
    preds = model.predict(data_scaled)

    if preprocessor is not None:
        data_scaled = preprocessor.transform(sample_input)
        preds = model.predict(data_scaled)
    else:
        preds = model.predict(sample_input)

    print(f"preds is {preds}")
    assert len(preds) == 1
    assert isinstance(preds[0], (int, float, np.integer, np.floating))
