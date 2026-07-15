"""Rolling-horizon MPC on top of a PINN surrogate (scipy SLSQP)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from marimo_flow.agents.schemas.control import ControlPlan

SurrogateFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Signature: ``surrogate(state_now, controls_over_horizon) -> state_trajectory``

Inputs:
  * ``state_now``: shape ``(n_states,)`` — current measurement/estimate.
  * ``controls_over_horizon``: shape ``(horizon, n_controls)``.
Returns:
  * ``state_trajectory``: shape ``(horizon, n_states)`` — predicted next states.
"""


def run_mpc_step(
    plan: ControlPlan,
    state_now: np.ndarray,
    surrogate: SurrogateFn,
    *,
    prev_controls: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve one MPC horizon; return the chosen control sequence + info.

    Only the first control column is applied by the caller — the rest
    is the MPC's look-ahead. ``prev_controls`` warm-starts the solver
    with the previous horizon shifted by one step.
    """
    h = plan.horizon
    n_ctrl = len(plan.controls)
    n_state = len(plan.states)
    if state_now.shape != (n_state,):
        raise ValueError(
            f"state_now must have shape ({n_state},); got {state_now.shape}"
        )

    lows = np.array([c.low for c in plan.controls])
    highs = np.array([c.high for c in plan.controls])
    x0 = _warm_start(plan, prev_controls)

    targets = np.array([s.target if s.target is not None else 0.0 for s in plan.states])
    weights = np.array([s.weight for s in plan.states])

    def flat_to_seq(x: np.ndarray) -> np.ndarray:
        return x.reshape(h, n_ctrl)

    def objective(x: np.ndarray) -> float:
        traj = surrogate(state_now, flat_to_seq(x))
        err = traj - targets
        return float(np.sum(weights * (err**2)))

    bounds = [
        (float(lows[i]), float(highs[i])) for _ in range(h) for i in range(n_ctrl)
    ]

    result = minimize(
        objective,
        x0=x0.flatten(),
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-6},
    )
    return flat_to_seq(result.x), {
        "cost": float(result.fun),
        "iterations": int(result.nit),
        "success": bool(result.success),
    }


def simulate_closed_loop(
    plan: ControlPlan,
    initial_state: np.ndarray,
    surrogate: SurrogateFn,
    true_dynamics: SurrogateFn,
    *,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Run n_steps of MPC + true-dynamics rollout; return the trajectory.

    ``true_dynamics`` plays the role of the real plant (for sim work).
    In production the caller replaces it with a measurement hook.
    """
    n_ctrl = len(plan.controls)
    state = initial_state.copy()
    applied = np.zeros((n_steps, n_ctrl))
    states = np.zeros((n_steps + 1, len(plan.states)))
    states[0] = state
    prev = None
    for k in range(n_steps):
        ctrl_seq, _ = run_mpc_step(plan, state, surrogate, prev_controls=prev)
        applied[k] = ctrl_seq[0]
        state = true_dynamics(state, ctrl_seq[:1])[0]
        states[k + 1] = state
        prev = ctrl_seq
    return {"states": states, "controls": applied}


def _warm_start(plan: ControlPlan, prev_controls: np.ndarray | None) -> np.ndarray:
    h, n = plan.horizon, len(plan.controls)
    if prev_controls is None:
        return np.tile(
            np.array([c.initial for c in plan.controls], dtype=float), (h, 1)
        )
    shifted = np.vstack([prev_controls[1:], prev_controls[-1:]])
    if shifted.shape != (h, n):
        return np.tile(
            np.array([c.initial for c in plan.controls], dtype=float), (h, 1)
        )
    return shifted
