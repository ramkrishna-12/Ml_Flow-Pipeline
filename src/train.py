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
