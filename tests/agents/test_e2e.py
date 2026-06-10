"""End-to-end smoke test — real MLflow file:// store, TestModel for all LLMs.

Drives the full sequence problem -> model -> solver -> training -> end with
the Managers + register_artifact stubbed so no torch work is needed, but
real MLflow artifacts are logged to verify persistence.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import pytest
from pydantic_ai.models.test import TestModel

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.runner import run_graph
from marimo_flow.agents.schemas import TaskSpec
from marimo_flow.agents.state import FlowState


@pytest.fixture
def tmp_mlflow():
    mlflow.set_experiment("agents-e2e")
    with mlflow.start_run() as run:
        yield run.info.run_id


def _make_fake_register_artifact(kind: str):
    """Write a small JSON artifact into the active MLflow run and return its URI."""

    def _fake(*, deps, state, artifact_path, filename, record, instance):
        run_id = state.mlflow_run_id
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / filename
            p.write_text(json.dumps({"stub": True, "kind": kind, **record}, indent=2))
            mlflow.MlflowClient().log_artifact(
                run_id, str(p), artifact_path=artifact_path
            )
        uri = f"runs:/{run_id}/{artifact_path}/{filename}"
        deps.registry[uri] = instance
        return uri

    return _fake


async def test_full_workflow_reaches_end(tmp_mlflow, monkeypatch):
    """problem -> model -> solver -> training -> end; all artifacts logged."""
    decisions = iter(
        [
            {"next_node": "problem", "rationale": "need problem first"},
            {"next_node": "model", "rationale": "need architecture"},
            {"next_node": "solver", "rationale": "wire solver"},
            {"next_node": "training", "rationale": "fit it"},
            {"next_node": "end", "rationale": "all done; solver trained"},
        ]
    )

    fake_model = object()
    fake_solver = object()
    fake_trainer = type("FT", (), {"callback_metrics": {"train_loss": 0.1}})()

    def fake_model_for(self, role):
        if role == "route":
            return TestModel(custom_output_args=next(decisions))
        if role == "problem":
            return TestModel(call_tools=["compose_problem"])
        if role in ("model", "solver"):
            return TestModel(call_tools=[f"build_{role}"])
        if role == "training":
            return TestModel(call_tools=["train"])
        return TestModel(call_tools=[])

    monkeypatch.setattr("marimo_flow.agents.deps.FlowDeps.model_for", fake_model_for)

    # Stub the composer and Manager.create + register_artifact for each
    # toolset module. Problem goes through the composer now, not a kind
    # dispatcher, so we stub compose_problem to return a fake class.
    fake_problem_cls = type("FakeProblem", (), {})

    def _fake_compose(spec):  # noqa: ARG001 — we ignore spec shape in the test
        return fake_problem_cls

    monkeypatch.setattr("marimo_flow.agents.toolsets.problem._compose", _fake_compose)
    # ProblemSpec validation still runs; TestModel auto-generates kwargs
    # for compose_problem and that dict won't parse as a ProblemSpec.
    # Monkeypatch ProblemSpec.model_validate to return a minimal valid spec.
    from marimo_flow.agents.schemas import ProblemSpec

    _valid_spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.problem.ProblemSpec.model_validate",
        classmethod(lambda cls, _data: _valid_spec),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.model.ModelManager.create",
        lambda kind, *, problem, **_kw: fake_model,
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.solver.SolverManager.create",
        lambda kind, *, problem, model, **_kw: fake_solver,
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.training.train_solver",
        lambda solver, **_kw: fake_trainer,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.problem.register_artifact",
        _make_fake_register_artifact("problem"),
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.model.register_artifact",
        _make_fake_register_artifact("model"),
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.solver.register_artifact",
        _make_fake_register_artifact("solver"),
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.training.register_artifact",
        _make_fake_register_artifact("training"),
    )

    # Pre-populate task_spec so TriageNode fast-paths to RouteNode
    # without needing a live triage LLM in the test.
    state = FlowState(
        user_intent="solve burgers 1d",
        mlflow_run_id=tmp_mlflow,
        task_spec=TaskSpec(
            title="Burgers 1D",
            description="Test e2e path",
            problem_kind="forward",
            equation_family="burgers",
            boundary_conditions=["u=0 on left and right walls"],
            initial_conditions=["u(x,0) = -sin(pi*x)"],
            material_properties={"viscosity": 0.01},
        ),
    )
    deps = FlowDeps(provenance_db_path=":memory:")

    result = await run_graph(state, deps, snapshot_run_id=tmp_mlflow)
    assert "done" in result.lower() or "trained" in result.lower()

    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(tmp_mlflow)}
    assert "problem" in artifacts
    assert "model" in artifacts
    assert "solver" in artifacts
    assert "training" in artifacts
    assert "agent_state" in artifacts
