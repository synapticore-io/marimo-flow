"""FunctionToolset for the Control agent (Phase F).

Wraps ``marimo_flow.control`` so the Control sub-agent can:

* build / validate a ``ControlPlan`` from free-form args;
* invoke ``run_mpc_step`` against a surrogate registered in
  ``deps.registry``;
* drive a closed-loop rollout for a demo.

The surrogate callable is supplied via ``deps.registry`` exactly like
the design toolset does its objective function — the Control agent
assembles a wrapper around the trained PINN and stores it under a
registry key before calling these tools.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import (
    ControlPlan,
    ControlVariableSpec,
    StateSpec,
)

control_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="control")


@control_toolset.tool
def build_control_plan(
    ctx: RunContext[FlowDeps],  # noqa: ARG001
    name: str,
    surrogate_uri: str,
    horizon: int,
    dt: float,
    controls: list[dict[str, Any]],
    states: list[dict[str, Any]],
    solver: str = "scipy_slsqp",
    objective_expression: str | None = None,
) -> dict[str, Any]:
    """Validate and return a serialised ControlPlan."""
    plan = ControlPlan(
        name=name,
        surrogate_uri=surrogate_uri,
        horizon=horizon,
        dt=dt,
        controls=[ControlVariableSpec(**c) for c in controls],
        states=[StateSpec(**s) for s in states],
        solver=solver,  # type: ignore[arg-type]
        objective_expression=objective_expression,
    )
    return plan.model_dump()


@control_toolset.tool
def mpc_step(
    ctx: RunContext[FlowDeps],
    plan: dict[str, Any],
    state_now: list[float],
    surrogate_registry_key: str,
) -> dict[str, Any]:
    """Solve one MPC horizon given a live surrogate callable."""
    from marimo_flow.control.mpc import run_mpc_step

    plan_model = ControlPlan.model_validate(plan)
    fn = ctx.deps.registry.get(surrogate_registry_key)
    if fn is None or not callable(fn):
        raise ModelRetry(
            f"no callable surrogate registered under {surrogate_registry_key!r}"
        )
    ctrl, info = run_mpc_step(plan_model, np.asarray(state_now, dtype=float), fn)
    return {"controls_horizon": ctrl.tolist(), **info}


@control_toolset.tool
def register_pinn_surrogate_tool(
    ctx: RunContext[FlowDeps],
    solver_registry_key: str,
    surrogate_registry_key: str,
    dt: float,
    output_field: str = "T",
    nx: int = 21,
    blend: float = 0.6,
) -> str:
    """Wrap a trained PINN solver as an MPC surrogate in ``deps.registry``."""
    from marimo_flow.control.pinn_surrogate import register_pinn_surrogate

    solver = ctx.deps.registry.get(solver_registry_key)
    if solver is None:
        raise ModelRetry(
            f"no solver registered under {solver_registry_key!r}; train first"
        )
    register_pinn_surrogate(
        ctx.deps.registry,
        surrogate_registry_key,
        solver,
        dt=dt,
        output_field=output_field,
        nx=nx,
        blend=blend,
    )
    return surrogate_registry_key


@control_toolset.tool
def closed_loop_simulation(
    ctx: RunContext[FlowDeps],
    plan: dict[str, Any],
    initial_state: list[float],
    surrogate_registry_key: str,
    true_dynamics_registry_key: str,
    n_steps: int = 20,
) -> dict[str, list[list[float]]]:
    """Run a closed-loop rollout and return the state / control trajectory."""
    from marimo_flow.control.mpc import simulate_closed_loop

    plan_model = ControlPlan.model_validate(plan)
    sur = ctx.deps.registry.get(surrogate_registry_key)
    plant = ctx.deps.registry.get(true_dynamics_registry_key)
    if not callable(sur) or not callable(plant):
        raise ModelRetry(
            "both surrogate and true-dynamics callables must be in deps.registry"
        )
    traj = simulate_closed_loop(
        plan_model,
        np.asarray(initial_state, dtype=float),
        sur,
        plant,
        n_steps=n_steps,
    )
    return {
        "states": traj["states"].tolist(),
        "controls": traj["controls"].tolist(),
    }
