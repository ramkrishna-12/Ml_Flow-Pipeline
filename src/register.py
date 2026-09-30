

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


def promote(best: dict, sp: Splits, s: Settings) -> dict:
    client = MlflowClient()
    run_id, m = best["run_id"], best["metrics"]

    failures = check_gates(m, s)
    if failures:
        raise QualityGateFailed("; ".join(failures))

    mv = mlflow.register_model(
        best["model_uri"],
        s.model_name,
        tags={"run_id": run_id, "gate": "passed", "test_pr_auc": f"{m['test_pr_auc']:.4f}"},
    )
    client.set_registered_model_alias(s.model_name, s.challenger_alias, mv.version)

    try:
        champ = client.get_model_version_by_alias(s.model_name, s.champion_alias)
    except MlflowException:
        champ = None

    if champ is None:
        promoted, reason, champ_pr = True, "no champion yet", None
    else:
        champ_m = evaluate_uri(f"models:/{s.model_name}@{s.champion_alias}", sp, s.decision_threshold)
        champ_pr = champ_m["test_pr_auc"]
        gain = m["test_pr_auc"] - champ_pr
        promoted = gain >= s.min_improvement
        reason = f"pr_auc gain {gain:+.4f} vs champion v{champ.version} (need >= {s.min_improvement})"

    if promoted:
        client.set_registered_model_alias(s.model_name, s.champion_alias, mv.version)
        client.delete_registered_model_alias(s.model_name, s.challenger_alias)
        client.set_model_version_tag(s.model_name, mv.version, "promoted", "true")

    return {
        "registered": True,
        "version": int(mv.version),
        "promoted": promoted,
        "reason": reason,
        "candidate_test_pr_auc": m["test_pr_auc"],
        "champion_test_pr_auc": champ_pr,
    }
