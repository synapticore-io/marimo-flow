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

# Surrogates are usually torch networks evaluated in float32. SLSQP's default
# finite-difference step (~1.5e-8) is below float32 resolution, so the
# objective would not change, the gradient would be zero and the solver would
# stop at the initial guess. sqrt(float32 eps) is the standard forward-
# difference step for float32-noisy functions.
_FD_REL_STEP = float(np.sqrt(np.finfo(np.float32).eps))


def run_mpc_step(
    plan: ControlPlan,
    state_now: np.ndarray,
    surrogate: SurrogateFn,
    *,
    prev_controls: np.ndarray | None = None,
    disturbance: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve one MPC horizon; return the chosen control sequence + info.

    Only the first control column is applied by the caller — the rest
    is the MPC's look-ahead. ``prev_controls`` warm-starts the solver
    with the previous horizon shifted by one step. ``disturbance`` (shape
    ``(n_states,)``) is added to every predicted state — the output
    disturbance estimate of offset-free MPC.
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
    offset = np.zeros(n_state) if disturbance is None else disturbance

    def flat_to_seq(x: np.ndarray) -> np.ndarray:
        return x.reshape(h, n_ctrl)

    def objective(x: np.ndarray) -> float:
        traj = surrogate(state_now, flat_to_seq(x)) + offset
        err = traj - targets
        return float(np.sum(weights * (err**2)))

    bounds = [
        (float(lows[i]), float(highs[i])) for _ in range(h) for i in range(n_ctrl)
    ]

    result = minimize(
        objective,
        x0=x0.flatten(),
        method="SLSQP",
        jac="2-point",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-6, "finite_diff_rel_step": _FD_REL_STEP},
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
    offset_free: bool = True,
    disturbance_gain: float = 0.3,
) -> dict[str, np.ndarray]:
    """Run n_steps of MPC + true-dynamics rollout; return the trajectory.

    ``true_dynamics`` plays the role of the real plant (for sim work).
    In production the caller replaces it with a measurement hook.

    With ``offset_free`` (default) an integrating output-disturbance
    observer corrects the surrogate: ``d += disturbance_gain * (measured -
    (surrogate + d))``, and every horizon is planned on ``surrogate + d``.
    A surrogate bias then leaves no steady-state error; ``d`` stays zero
    when the surrogate matches the plant. A gain below 1 keeps the estimate
    from chasing input-dependent (e.g. gain) model errors step by step,
    which would make the loop oscillate.
    Returns ``states`` and ``disturbance`` of shape ``(n_steps + 1,
    n_states)`` and ``controls`` of shape ``(n_steps, n_controls)``.
    """
    n_ctrl = len(plan.controls)
    n_state = len(plan.states)
    state = initial_state.copy()
    applied = np.zeros((n_steps, n_ctrl))
    states = np.zeros((n_steps + 1, n_state))
    disturbances = np.zeros((n_steps + 1, n_state))
    states[0] = state
    prev = None
    d = np.zeros(n_state)
    for k in range(n_steps):
        ctrl_seq, _ = run_mpc_step(
            plan, state, surrogate, prev_controls=prev, disturbance=d
        )
        applied[k] = ctrl_seq[0]
        predicted = surrogate(state, ctrl_seq[:1])[0] + d
        state = true_dynamics(state, ctrl_seq[:1])[0]
        if offset_free:
            d = d + disturbance_gain * (state - predicted)
        states[k + 1] = state
        disturbances[k + 1] = d
        prev = ctrl_seq
    return {"states": states, "controls": applied, "disturbance": disturbances}


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
