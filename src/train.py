import os
import subprocess
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, Settings

# MLflow 3.x serializes sklearn models with SKOPS (safe: no arbitrary code on load, unlike pickle).
# Tree ensembles need their node-storage type explicitly allow-listed. Review before adding to it.
SKOPS_TRUSTED = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

# Candidate configs for the sweep. Deliberately small so CI stays fast.
SWEEP = [
    ("logreg", {"C": 0.1}),
    ("logreg", {"C": 1.0}),
    ("rf", {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 5}),
    ("hgb", {"learning_rate": 0.08, "max_depth": 4, "max_iter": 150}),
]
DEFAULT_SINGLE = [("rf", {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 5})]


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def split_data(df: pd.DataFrame, s: Settings) -> Splits:
    """train / val / test. Val picks the best candidate; test is touched only for the
    final report and the promotion gate -> no leakage from model selection."""
    X = df[FEATURES].astype("float64")  # all-double => simple, strict signature
    y = df[s.label_col].astype(int)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=s.test_size, stratify=y, random_state=s.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=s.seed
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


def build_pipeline(model_type: str, params: dict, seed: int) -> Pipeline:
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", **params)
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_jobs=-1, class_weight="balanced_subsample", random_state=seed, **params
        )
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"unknown model_type {model_type}")
    # Scaler lives INSIDE the pipeline -> the logged artifact is the whole preprocessing+model,
    # so training/serving skew is impossible by construction.
    pre = ColumnTransformer([("scale", StandardScaler(), FEATURES)])
    return Pipeline([("pre", pre), ("clf", clf)])



def compute_metrics(y_true, proba, threshold: float = 0.5, prefix: str = "") -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    m = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),  # the metric that matters at 3% fraud
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    return {f"{prefix}{k}": float(v) for k, v in m.items()}


def _git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _run_tags() -> dict:
    tags = {"git_sha": _git_sha(), "trigger": os.getenv("GITHUB_EVENT_NAME", "manual")}
    if os.getenv("GITHUB_RUN_ID"):
        tags["ci_run_url"] = (
            f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
    return tags


def train_one(model_type: str, params: dict, sp: Splits, s: Settings, dataset=None, nested=False):
    """One candidate = one MLflow run. Returns {run_id, model_type, metrics}."""
    with mlflow.start_run(run_name=f"{model_type}", nested=nested) as run:
        mlflow.set_tags({**_run_tags(), "model_type": model_type, "stage": "candidate"})
        mlflow.log_params({"model_type": model_type, **params, "threshold": s.decision_threshold})
        mlflow.log_params({"n_train": len(sp.X_train), "fraud_rate_train": round(sp.y_train.mean(), 4)})
        if dataset is not None:
            mlflow.log_input(dataset, context="training")  # dataset lineage

        pipe = build_pipeline(model_type, params, s.seed)
        pipe.fit(sp.X_train, sp.y_train)

        val_p = pipe.predict_proba(sp.X_val)[:, 1]
        test_p = pipe.predict_proba(sp.X_test)[:, 1]
        metrics = {
            **compute_metrics(sp.y_val, val_p, s.decision_threshold, "val_"),
            **compute_metrics(sp.y_test, test_p, s.decision_threshold, "test_"),
        }
        mlflow.log_metrics(metrics)

        # ---- artifacts: figures + permutation importance ----
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(sp.y_test, test_p, ax=ax)
        mlflow.log_figure(fig, "plots/pr_curve.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            sp.y_test, (test_p >= s.decision_threshold).astype(int), ax=ax
        )
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)
        imp = permutation_importance(
            pipe,
            sp.X_test,
            sp.y_test,
            scoring="average_precision",
            n_repeats=3,
            random_state=s.seed,
            n_jobs=1,
        )
        mlflow.log_dict(
            dict(sorted(zip(FEATURES, map(float, imp.importances_mean), strict=True), key=lambda kv: -kv[1])),
            "feature_importance.json",
        )

        # ---- the model itself ----
        sig_out = pipe.predict_proba(sp.X_train.head(50))
        signature = infer_signature(sp.X_train.head(50), sig_out)
        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=sp.X_train.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED,
        )
        return {
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,  # MLflow 3: models:/m-<id> (a LoggedModel), not runs:/.../model
            "model_type": model_type,
            "metrics": metrics,
        }


def run_sweep(sp: Splits, s: Settings, data_stats: dict, candidates=None, dataset=None) -> dict:
    """Parent run 'sweep' with one nested child per candidate. Picks best by VALIDATION pr_auc."""
    candidates = candidates or SWEEP
    with mlflow.start_run(run_name="sweep") as parent:
        mlflow.set_tags({**_run_tags(), "stage": "sweep"})
        mlflow.log_params({f"data_{k}": v for k, v in data_stats.items()})
        mlflow.log_param("n_candidates", len(candidates))
        results = [train_one(m, p, sp, s, dataset, nested=True) for m, p in candidates]
        best = max(results, key=lambda r: r["metrics"]["val_pr_auc"])
        mlflow.set_tag("best_run_id", best["run_id"])
        mlflow.log_metric("best_val_pr_auc", best["metrics"]["val_pr_auc"])
        mlflow.log_metric("best_test_pr_auc", best["metrics"]["test_pr_auc"])
        best["parent_run_id"] = parent.info.run_id
        return best
