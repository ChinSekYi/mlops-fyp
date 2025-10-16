"""
Simple test to verify shared fixtures are working correctly.
"""

import pandas as pd


def test_mock_raw_df_fixture(mock_raw_df):
    """Test that the shared mock_raw_df fixture works correctly."""

    # Verify it's a DataFrame
    assert isinstance(mock_raw_df, pd.DataFrame)

    # Verify it has the expected structure
    assert len(mock_raw_df) == 50  # 50 rows (updated for pipeline integration tests)
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

    # Verify expected keys exist (RAW format)
    expected_keys = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type",
        "nameOrig",  # Raw account ID string
        "nameDest",  # Raw account ID string
    ]

    for key in expected_keys:
        assert key in sample_input, f"Missing key: {key}"

    # Verify data types
    assert isinstance(sample_input["step"], int)
    assert isinstance(sample_input["amount"], float)
    assert isinstance(sample_input["type"], str)
    assert isinstance(
        sample_input["nameOrig"], str
    )  # Should be string like "C84071102"
    assert isinstance(
        sample_input["nameDest"], str
    )  # Should be string like "C1576697216"

    print("✅ sample_input fixture is working correctly!")


def test_fixtures_consistency(mock_raw_df, sample_input):
    """Test that fixtures are consistent with each other."""

    # Both fixtures should use the same RAW format
    # mock_raw_df represents raw data, sample_input represents raw API input

    # Check that both have raw name columns (not tokenized)
    assert "nameOrig" in mock_raw_df.columns
    assert "nameDest" in mock_raw_df.columns
    assert "nameOrig" in sample_input
    assert "nameDest" in sample_input

    # Check that both use the same transaction types
    valid_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    assert sample_input["type"] in valid_types
    assert all(t in valid_types for t in mock_raw_df["type"].unique())

    # Check data consistency - both should have same structure
    common_fields = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type",
    ]

    for field in common_fields:
        assert field in sample_input, f"sample_input missing field: {field}"
        assert field in mock_raw_df.columns, f"mock_raw_df missing column: {field}"

    # Check account ID format consistency (should be strings starting with C/M)
    assert isinstance(sample_input["nameOrig"], str)
    assert isinstance(sample_input["nameDest"], str)
    assert all(isinstance(x, str) for x in mock_raw_df["nameOrig"])
    assert all(isinstance(x, str) for x in mock_raw_df["nameDest"])

    print("✅ Fixtures are consistent with raw data format!")
