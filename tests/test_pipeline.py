import mlflow
import pytest
from mlflow import MlflowClient

from src import pipeline, register
from src.config import get_settings


def test_pipeline_registers_and_sets_champion(env):
    res = pipeline.run(sweep=False, rows=4000, out=str(env / "res.json"))
    s = get_settings()
    assert res["registered"] and res["promoted"] and res["version"] == 1
    mv = MlflowClient().get_model_version_by_alias(s.model_name, s.champion_alias)
    assert str(mv.version) == "1"

    # run name + nested structure exist and metrics were logged
    exp = mlflow.get_experiment_by_name(s.experiment_name)
    runs = mlflow.search_runs([exp.experiment_id])
    assert {"sweep", "rf"} <= set(runs["tags.mlflow.runName"])
    assert "metrics.test_pr_auc" in runs.columns


def test_second_identical_run_is_not_promoted(env):
    pipeline.run(sweep=False, rows=4000, out=str(env / "a.json"))
    res = pipeline.run(sweep=False, rows=4000, out=str(env / "b.json"))
    assert res["registered"] and not res["promoted"]
    s = get_settings()
    assert str(MlflowClient().get_model_version_by_alias(s.model_name, s.champion_alias).version) == "1"


def test_quality_gate_blocks_registration(env, monkeypatch):
    monkeypatch.setenv("MIN_ROC_AUC", "0.999")
    with pytest.raises(register.QualityGateFailed):
        pipeline.run(sweep=False, rows=4000, out=str(env / "c.json"))
    s = get_settings()
    assert MlflowClient().search_registered_models(filter_string=f"name='{s.model_name}'") == []
