
import argparse
import json
import os
import sys

import mlflow
from mlflow import MlflowClient

from src import data, register, train
from src.config import get_settings


def run(sweep: bool = True, rows: int = 20_000, out: str = "artifacts/pipeline_result.json") -> dict:
    s = get_settings()
    mlflow.set_tracking_uri(s.tracking_uri)
    mlflow.set_experiment(s.experiment_name)

    df = data.load_or_generate(s.data_source, n=rows, seed=s.seed)
    stats = data.validate(df, s.label_col)
    print(f"[data] {stats}")

    sp = train.split_data(df, s)
    dataset = mlflow.data.from_pandas(df, source=s.data_source, name="transactions", targets=s.label_col)

    best = train.run_sweep(sp, s, stats, candidates=None if sweep else train.DEFAULT_SINGLE, dataset=dataset)
    pr = best["metrics"]["test_pr_auc"]
    print(f"[train] best={best['model_type']} run={best['run_id']} test_pr_auc={pr:.4f}")

    client = MlflowClient()
    try:
        decision = register.promote(best, sp, s)
    except register.QualityGateFailed as e:
        client.set_tag(best["parent_run_id"], "promotion_decision", f"GATE FAILED: {e}")
        result = {"registered": False, "promoted": False, "reason": f"gate failed: {e}"}
        _write(out, result)
        print(f"[gate] FAILED: {e}")
        raise
    client.set_tag(best["parent_run_id"], "promotion_decision", json.dumps(decision))
    result = {
        **decision,
        "run_id": best["run_id"],
        "model_type": best["model_type"],
        "tracking_uri": s.tracking_uri,
        "model_name": s.model_name,
    }
    _write(out, result)
    print(f"[registry] {result}")
    return result


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--out", default="artifacts/pipeline_result.json")
    a = ap.parse_args()
    try:
        run(sweep=not a.no_sweep, rows=a.rows, out=a.out)
    except register.QualityGateFailed:
        sys.exit(2)


if __name__ == "__main__":
    main()
