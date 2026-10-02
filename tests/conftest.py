import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated MLflow (sqlite + local artifacts) and data path per test. Lenient gates: tiny data."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-exp")
    monkeypatch.setenv("DATA_SOURCE", str(tmp_path / "tx.parquet"))
    monkeypatch.setenv("MIN_ROC_AUC", "0.6")
    monkeypatch.setenv("MIN_PR_AUC", "0.05")
    return tmp_path
