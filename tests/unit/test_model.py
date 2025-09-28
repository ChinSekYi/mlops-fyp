def test_model_load(model):
    """Check that the model can be loaded."""
    assert model is not None


def test_model_prediction_shape(model):
    """Check that model returns correct output length on dummy input."""
    sample_row = [[0.0] * 30]
    preds = model.predict(sample_row)
    assert len(preds) == 1
    assert isinstance(preds[0], (int, float))
