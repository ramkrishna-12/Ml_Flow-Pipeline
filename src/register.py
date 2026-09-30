import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import Settings
from src.train import Splits, compute_metrics


class QualityGateFailed(RuntimeError):
    pass


def positive_proba(pred) -> np.ndarray:
    a = np.asarray(pred)
    return a[:, 1] if a.ndim == 2 else a


def evaluate_uri(model_uri: str, sp: Splits, threshold: float) -> dict:
    """Score ANY registered model on the SAME holdout -> apples-to-apples comparison."""
    model = mlflow.pyfunc.load_model(model_uri)
    return compute_metrics(sp.y_test, positive_proba(model.predict(sp.X_test)), threshold, "test_")


def check_gates(metrics: dict, s: Settings) -> list[str]:
    failures = []
    if metrics["test_roc_auc"] < s.min_roc_auc:
        failures.append(f"roc_auc {metrics['test_roc_auc']:.3f} < {s.min_roc_auc}")
    if metrics["test_pr_auc"] < s.min_pr_auc:
        failures.append(f"pr_auc {metrics['test_pr_auc']:.3f} < {s.min_pr_auc}")
    return failures
