"""
Simple test to verify shared fixtures are working correctly.
"""

import pandas as pd


def test_mock_raw_df_fixture(mock_raw_df):
    """Test that the shared mock_raw_df fixture works correctly."""

    # Verify it's a DataFrame
    assert isinstance(mock_raw_df, pd.DataFrame)

    # Verify it has the expected structure
    assert len(mock_raw_df) == 12  # 12 rows
    assert len(mock_raw_df.columns) >= 10  # At least 10 columns

    # Verify expected columns exist
    expected_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type",
        "nameOrig",
        "nameDest",
        "isFlaggedFraud",
        "isFraud",
    ]

    for col in expected_columns:
        assert col in mock_raw_df.columns, f"Missing column: {col}"

    # Verify target column has expected values
    assert "isFraud" in mock_raw_df.columns
    assert mock_raw_df["isFraud"].nunique() == 2  # Should have 0 and 1

    # Verify data types make sense
    assert mock_raw_df["amount"].dtype in ["float64", "int64"]
    assert mock_raw_df["type"].dtype == "object"  # String column


def test_sample_input_fixture(sample_input):
    """Test that the sample_input fixture works correctly."""

    # Verify it's a dictionary
    assert isinstance(sample_input, dict)

    # Verify expected keys exist
    expected_keys = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type",
        "nameOrig_token",
        "nameDest_token",
    ]

    for key in expected_keys:
        assert key in sample_input, f"Missing key: {key}"

    # Verify data types
    assert isinstance(sample_input["step"], int)
    assert isinstance(sample_input["amount"], float)
    assert isinstance(sample_input["type"], str)

    print("✅ sample_input fixture is working correctly!")


def test_fixtures_consistency(mock_raw_df, sample_input):
    """Test that fixtures are consistent with each other."""

    # The sample_input should have the same structure as processed data from mock_raw_df
    # After transformation, nameOrig and nameDest become nameOrig_token and nameDest_token

    # Check that sample_input has tokenized versions of name columns
    assert "nameOrig_token" in sample_input
    assert "nameDest_token" in sample_input

    # Check that mock_raw_df has the original name columns
    assert "nameOrig" in mock_raw_df.columns
    assert "nameDest" in mock_raw_df.columns

    # Check data consistency
    assert isinstance(
        sample_input["amount"], type(float(mock_raw_df["amount"].iloc[0]))
    )
