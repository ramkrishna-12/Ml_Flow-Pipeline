
import os
from dataclasses import dataclass, field


def _env(name: str, default, cast=str):
    return field(default_factory=lambda: cast(os.getenv(name, default)))


@dataclass(frozen=True)
class Settings:
    # --- MLflow ---
    tracking_uri: str = _env("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name: str = _env("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
    model_name: str = _env("MODEL_NAME", "fraud-detector")
    champion_alias: str = _env("CHAMPION_ALIAS", "champion")
    challenger_alias: str = _env("CHALLENGER_ALIAS", "challenger")

    # --- Data --- (local path or s3://bucket/key.parquet)
    data_source: str = _env("DATA_SOURCE", "data/transactions.parquet")
    label_col: str = "is_fraud"
    test_size: float = _env("TEST_SIZE", 0.2, float)
    seed: int = _env("SEED", 42, int)

    # --- Quality gates: a candidate must clear ALL of these to even be registered ---
    min_roc_auc: float = _env("MIN_ROC_AUC", 0.85, float)
    min_pr_auc: float = _env("MIN_PR_AUC", 0.30, float)
    # challenger must beat the champion's PR-AUC by at least this much to be promoted
    min_improvement: float = _env("MIN_IMPROVEMENT", 0.002, float)

    # --- Serving ---
    decision_threshold: float = _env("DECISION_THRESHOLD", 0.5, float)
