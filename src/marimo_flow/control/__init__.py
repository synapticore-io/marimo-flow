"""Closed-loop control on top of a trained PINN surrogate.

Scope (Phase F):

* :func:`run_mpc_step` — single rolling-horizon optimisation given a
  state estimate, a ``ControlPlan``, and a callable surrogate.
* :func:`simulate_closed_loop` — drive a toy dynamics model forward
  under the MPC's selected controls so the demo notebook can report a
  closed-loop trajectory.

Kept deliberately small — scipy.optimize SLSQP is the inner QP engine,
no casadi / cvxpy / do-mpc. For anything beyond scalar-dynamics
examples escalate to Lead.
"""

from marimo_flow.control.heat_rod import (
    FiniteDifferenceHeatRod,
    build_heat_rod_problem_spec,
    generate_step_dataset,
    make_plant_stepper,
    make_rollout_surrogate,
    train_step_surrogate,
)
from marimo_flow.control.mpc import run_mpc_step, simulate_closed_loop
from marimo_flow.control.pinn_surrogate import (
    make_pinn_rollout_surrogate,
    query_field_mean,
    register_pinn_surrogate,
)

__all__ = [
    "FiniteDifferenceHeatRod",
    "build_heat_rod_problem_spec",
    "generate_step_dataset",
    "make_pinn_rollout_surrogate",
    "make_plant_stepper",
    "make_rollout_surrogate",
    "query_field_mean",
    "register_pinn_surrogate",
    "run_mpc_step",
    "simulate_closed_loop",
    "train_step_surrogate",
]
