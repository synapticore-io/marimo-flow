"""Verify the local MLflow layout: data/mlflow/{db,artifacts}, no ./mlruns/."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest


@pytest.fixture(autouse=True)
def _reset_lead_state(monkeypatch):
    """Force _ensure_tracking_uri / _ensure_local_layout to redo their work
    inside this test's tmp CWD (the autouse conftest sets a session-scoped
    sqlite URI, which would otherwise short-circuit the layout helper)."""
    import marimo_flow.agents.lead as _lead

    monkeypatch.setattr(_lead, "_TRACKING_URI_APPLIED", None, raising=False)
    monkeypatch.setattr(_lead, "_LAYOUT_PREPARED", None, raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)


def test_default_layout_uses_data_mlflow_no_mlruns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from marimo_flow.agents.lead import _ensure_tracking_uri

    _ensure_tracking_uri()

    # 1. The DB sits under data/mlflow/db/.
    assert (tmp_path / "data" / "mlflow" / "db" / "mlflow.db").exists()
    # 2. The artifacts directory was created.
    assert (tmp_path / "data" / "mlflow" / "artifacts").is_dir()
    # 3. CWD does NOT contain ./mlruns/ — that's the legacy Default-Experiment
    #    fallback we're moving away from.
    assert not (tmp_path / "mlruns").exists(), (
        f"./mlruns/ leaked: {sorted(p.name for p in tmp_path.iterdir())}"
    )

    # 4. A logged run lands under data/mlflow/artifacts/<run_id>/artifacts.
    with mlflow.start_run(run_name="layout-check") as run:
        mlflow.log_metric("loss", 0.123)

    art_uri = run.info.artifact_uri.replace("\\", "/")
    assert "data/mlflow/artifacts" in art_uri, art_uri

    # 5. The marimo-flow experiment was registered (not the Default one).
    exp = mlflow.get_experiment(run.info.experiment_id)
    assert exp.name == "marimo-flow"
    assert exp.artifact_location.replace("\\", "/").endswith("/data/mlflow/artifacts")


def test_remote_uri_skips_local_layout(tmp_path, monkeypatch):
    """A non-default tracking URI must not create data/mlflow/ on the host."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.invalid:5000")

    from marimo_flow.agents.lead import _ensure_tracking_uri

    # We don't actually log a run (server is unreachable), we just want to
    # observe that the layout helper bowed out.
    _ensure_tracking_uri()
    assert not (tmp_path / "data" / "mlflow").exists()
    assert mlflow.get_tracking_uri() == "http://mlflow.invalid:5000"


def test_explicit_local_db_path_also_triggers_layout(tmp_path, monkeypatch):
    """An absolute SQLite URI ending in data/mlflow/db/mlflow.db is still
    recognised as the local default and gets the layout treatment."""
    monkeypatch.chdir(tmp_path)
    target = (tmp_path / "data" / "mlflow" / "db" / "mlflow.db").resolve()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{target.as_posix()}")

    from marimo_flow.agents.lead import _ensure_tracking_uri

    _ensure_tracking_uri()
    assert (tmp_path / "data" / "mlflow" / "artifacts").is_dir()
    assert mlflow.get_experiment_by_name("marimo-flow") is not None


def test_custom_experiment_name_via_env(tmp_path, monkeypatch):
    """``MLFLOW_EXPERIMENT_NAME`` overrides the default experiment name."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "my-project")

    from marimo_flow.agents.lead import _ensure_tracking_uri

    _ensure_tracking_uri()
    exp = mlflow.get_experiment_by_name("my-project")
    assert exp is not None
    assert Path(exp.artifact_location.replace("file:///", "")).name == "artifacts"
