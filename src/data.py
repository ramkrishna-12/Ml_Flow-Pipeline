
import io
import os
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd

from src.config import FEATURES


class DataValidationError(ValueError):
    pass


def generate_synthetic(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "amount": rng.lognormal(mean=3.5, sigma=1.1, size=n).round(2),
            "hour": rng.integers(0, 24, size=n),
            "merchant_risk": rng.beta(2, 8, size=n).round(4),
            "distance_from_home_km": rng.exponential(scale=15, size=n).round(2),
            "txn_count_24h": rng.poisson(lam=2.5, size=n),
            "account_age_days": rng.integers(1, 3650, size=n),
            "is_foreign": rng.binomial(1, 0.08, size=n),
        }
    )
    # Latent fraud score -> label. Signal is real but noisy (ROC-AUC ~0.9, fraud rate ~3%).
    z = (
        -7.0
        + 0.010 * df["amount"]
        + 6.0 * df["merchant_risk"]
        + 0.03 * df["distance_from_home_km"]
        + 0.35 * df["txn_count_24h"]
        + 1.6 * df["is_foreign"]
        + 1.2 * df["hour"].isin([0, 1, 2, 3, 4]).astype(int)
        - 0.0004 * df["account_age_days"]
        + 0.000025 * df["amount"] * df["distance_from_home_km"]  # interaction: big + far-away
        + rng.normal(0, 0.4, size=n)
    ) * 1.3
    p = 1 / (1 + np.exp(-z))
    df["is_fraud"] = rng.binomial(1, p)
    return df


def read_dataset(source: str) -> pd.DataFrame:
    if source.startswith("s3://"):
        bucket, key = _split_s3(source)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    return pd.read_parquet(source)


def write_dataset(df: pd.DataFrame, dest: str) -> None:
    if dest.startswith("s3://"):
        bucket, key = _split_s3(dest)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    else:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        df.to_parquet(dest, index=False)


def load_or_generate(source: str, n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Read `source`; if it does not exist yet, generate + persist it (bootstraps a fresh env)."""
    try:
        return read_dataset(source)
    except Exception:  # FileNotFoundError locally, NoSuchKey on S3
        df = generate_synthetic(n, seed)
        write_dataset(df, source)
        return df