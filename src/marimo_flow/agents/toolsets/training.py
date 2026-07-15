"""FunctionToolset for the Training agent.

Runs `core.train_solver` on the registered solver, logs the resulting
training run to MLflow, and stashes the fitted Trainer in the registry.
"""

from __future__ import annotations

import contextlib
from typing import Any

import mlflow
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.toolsets._registry import register_artifact, require_state
from marimo_flow.agents.toolsets._specs import run_config_from
from marimo_flow.core import train_solver

training_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="training")


@training_toolset.tool
def discretise_domain(
    ctx: RunContext[FlowDeps],
    n: int = 1000,
    mode: str = "random",
) -> str:
    """Sample collocation points on the registered problem's domain.

    Args:
        n: Number of points. Typical values: 200 for quick sanity checks,
           1000–10000 for proper training.
        mode: 'grid' (uniform, test), 'random' (Monte Carlo, default),
              'lh' (Latin Hypercube).
    """
    state = require_state(ctx.deps)
    if state.problem_artifact_uri is None:
        raise ModelRetry(
            "No problem registered yet. Hand control back so the problem "
            "agent can run first, then retry discretise_domain."
        )
    problem = ctx.deps.registry[state.problem_artifact_uri]
    problem.discretise_domain(n=n, mode=mode)
    return f"Discretised domain: n={n}, mode={mode!r}."


@training_toolset.tool
def train(
    ctx: RunContext[FlowDeps],
    max_epochs: int = 1000,
    accelerator: str = "auto",
    n_points: int = 1000,
    sample_mode: str = "random",
) -> dict[str, Any]:
    """Fit the registered solver via pina.Trainer.

    Starts an MLflow nested run named 'training' and enables
    `mlflow.pytorch.autolog()` (already on) so pytorch-Lightning metrics
    and checkpoints are captured automatically.

    Returns a dict with final loss, run_id, and a short summary string.
    """
    state = require_state(ctx.deps)
    if state.solver_artifact_uri is None:
        raise ModelRetry(
            "No solver registered yet. Hand control back so the solver "
            "agent can run first, then retry train."
        )
    solver = ctx.deps.registry[state.solver_artifact_uri]

    with mlflow.start_run(nested=True, run_name="training") as run:
        training_run_id = run.info.run_id
        trainer = train_solver(
            solver,
            max_epochs=max_epochs,
            accelerator=accelerator,
            n_points=n_points,
            sample_mode=sample_mode,
        )
        final_loss = None
        if trainer.callback_metrics:
            for key in ("train_loss", "loss", "total_loss"):
                if key in trainer.callback_metrics:
                    final_loss = float(trainer.callback_metrics[key])
                    break

    uri = register_artifact(
        deps=ctx.deps,
        state=state,
        artifact_path="training",
        filename="training_spec.json",
        record={
            "max_epochs": max_epochs,
            "accelerator": accelerator,
            "n_points": n_points,
            "sample_mode": sample_mode,
            "training_run_id": training_run_id,
            "final_loss": final_loss,
        },
        instance=trainer,
    )
    state.training_artifact_uri = uri
    state.training_run_id = training_run_id
    cfg = run_config_from(
        max_epochs=max_epochs,
        accelerator=accelerator,
        n_points=n_points,
        sample_mode=sample_mode,
    )
    state.run_config = cfg
    with contextlib.suppress(Exception):
        task_id = state.task_spec.task_id if state.task_spec else "unknown"
        ctx.deps.provenance().record_run_config(task_id, cfg)
        if final_loss is not None:
            ctx.deps.provenance().record_metric(
                experiment_id=None,
                run_id=training_run_id,
                name="final_loss",
                value=float(final_loss),
            )
    return {
        "training_run_id": training_run_id,
        "final_loss": final_loss,
        "uri": uri,
        "summary": f"Trained {type(solver).__name__} for {max_epochs} epochs."
        + (f" Final loss: {final_loss:.6g}." if final_loss is not None else ""),
    }
