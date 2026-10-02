import boto3
import pytest
from moto import mock_aws

from src import data


def test_generate_shape_and_imbalance():
    df = data.generate_synthetic(5000, seed=1)
    assert len(df) == 5000
    assert 0.01 < df["is_fraud"].mean() < 0.10  # imbalanced by design


def test_validate_ok():
    assert data.validate(data.generate_synthetic(5000))["rows"] == 5000


def test_validate_rejects_nulls_and_missing_cols():
    df = data.generate_synthetic(5000)
    bad = df.copy()
    bad.loc[0, "amount"] = None
    with pytest.raises(data.DataValidationError, match="nulls"):
        data.validate(bad)
    with pytest.raises(data.DataValidationError, match="missing"):
        data.validate(df.drop(columns=["hour"]))


@mock_aws
def test_s3_roundtrip(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    boto3.client("s3").create_bucket(Bucket="test-bucket")
    df = data.generate_synthetic(2000)
    data.write_dataset(df, "s3://test-bucket/data/tx.parquet")
    assert data.read_dataset("s3://test-bucket/data/tx.parquet").shape == df.shape
