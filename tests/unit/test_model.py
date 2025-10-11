"""
Unit tests for the Model
Tests model load, shape

"""

import numpy as np
import pandas as pd


def test_model_load(model):
    """Check that the model can be loaded."""
    assert model is not None


def test_model_prediction_shape(model):
    """Check that model returns correct output length on dummy input."""
    # PaySim dataset after preprocessing has 13 features:
    # step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
    # type__CASH_IN, type__CASH_OUT, type__DEBIT, type__PAYMENT, type__TRANSFER,
    # nameOrig_token, nameDest_token
    sample_input = pd.DataFrame(
        {
            "step": [1],
            "amount": [1000.0],
            "oldbalanceOrg": [5000.0],
            "newbalanceOrig": [4000.0],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [1000.0],
            "type__CASH_IN": [0],
            "type__CASH_OUT": [1],
            "type__DEBIT": [0],
            "type__PAYMENT": [0],
            "type__TRANSFER": [0],
            "nameOrig_token": [123],
            "nameDest_token": [456],
        }
    )
    preds = model.predict(sample_input)
    print(f"preds is {preds}")  # preds is [0]
    assert len(preds) == 1
    assert isinstance(preds[0], (int, float, np.integer, np.floating))
