"""FunctionToolset for the lead agent's `run_pina_workflow` tool.

This tool kicks off the `pydantic-graph` state machine with the sub-agent
team (TriageNode → RouteNode → Notebook/Problem/Model/Solver/Validation/MLflow).
It shares `deps` with the sub-agents so they all see the same FlowDeps +
FlowState instance.

Every invocation is wrapped in an ``ExperimentRecord`` that is opened
before the graph run (status=``running``) and closed afterwards with
a status inferred from the final FlowState (completed / failed).
"""

from __future__ import annotations

import mlflow
from pydantic_ai import FunctionToolset, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.runner import run_graph
from marimo_flow.agents.services import complete_experiment, start_experiment
from marimo_flow.agents.state import FlowState

lead_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="lead")


@lead_toolset.tool
async def run_pina_workflow(ctx: RunContext[FlowDeps], intent: str) -> str:
    """Run the PINA team graph end-to-end. Returns the team's final summary."""
    if mlflow.active_run() is None:
        run_id = mlflow.start_run().info.run_id
    else:
        run_id = mlflow.active_run().info.run_id
    state = FlowState(user_intent=intent, mlflow_run_id=run_id)
    ctx.deps.state = state
    experiment = start_experiment(state, ctx.deps)
    try:
        output = await run_graph(state, ctx.deps, snapshot_run_id=run_id)
    except Exception:
        complete_experiment(experiment, state, ctx.deps, status="failed")
        raise
    complete_experiment(experiment, state, ctx.deps)
    return output
