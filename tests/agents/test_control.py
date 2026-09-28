"""Tests for the MPC control layer — pure-Python surrogate, no PINN."""

from __future__ import annotations

import numpy as np

from marimo_flow.agents.schemas import (
    ControlPlan,
    ControlVariableSpec,
    StateSpec,
)
from marimo_flow.control import run_mpc_step, simulate_closed_loop


def _scalar_surrogate(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Toy linear dynamics: x_{k+1} = 0.9 x_k + u_k."""
    traj = np.zeros((len(controls), 1))
    x = float(state[0])
    for i, u in enumerate(controls[:, 0]):
        x = 0.9 * x + float(u)
        traj[i, 0] = x
    return traj


def test_mpc_step_drives_state_toward_target():
    plan = ControlPlan(
        name="thermostat",
        surrogate_uri="mem://scalar",
        horizon=5,
        dt=1.0,
        controls=[ControlVariableSpec(name="u", low=-1.0, high=1.0)],
        states=[StateSpec(name="x", target=0.0, weight=1.0)],
    )
    state_now = np.array([2.0])
    ctrl, info = run_mpc_step(plan, state_now, _scalar_surrogate)
    assert ctrl.shape == (5, 1)
    # Cost reduces the state toward 0 — first control should be negative.
    assert ctrl[0, 0] <= 0.0
    assert info["success"]


def _float32_surrogate(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Same toy dynamics, computed in float32 like a torch/PINN surrogate."""
    traj = np.zeros((len(controls), 1))
    x = np.float32(state[0])
    for i, u in enumerate(controls[:, 0]):
        x = np.float32(0.5) * x + np.float32(0.5) * np.float32(u)
        traj[i, 0] = x
    return traj


def test_mpc_step_moves_controls_with_float32_surrogate():
    # SLSQP's default finite-difference step (~1.5e-8) is below float32
    # resolution: the objective does not change, the gradient is zero and
    # the solver stops at the initial guess.
    plan = ControlPlan(
        name="float32",
        surrogate_uri="mem://f32",
        horizon=4,
        dt=1.0,
        controls=[ControlVariableSpec(name="u", low=0.0, high=1.0, initial=0.5)],
        states=[StateSpec(name="x", target=0.9, weight=1.0)],
    )
    ctrl, info = run_mpc_step(plan, np.array([0.0]), _float32_surrogate)
    assert info["iterations"] > 1
    assert ctrl[0, 0] > 0.9


def _biased_surrogate(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Plant model that under-predicts: believes u only half as effective."""
    traj = np.zeros((len(controls), 1))
    x = float(state[0])
    for i, u in enumerate(controls[:, 0]):
        x = 0.5 * x + 0.25 * float(u)
        traj[i, 0] = x
    return traj


def _true_plant(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    traj = np.zeros((len(controls), 1))
    x = float(state[0])
    for i, u in enumerate(controls[:, 0]):
        x = 0.5 * x + 0.5 * float(u)
        traj[i, 0] = x
    return traj


def _mismatch_plan() -> ControlPlan:
    return ControlPlan(
        name="mismatch",
        surrogate_uri="mem://biased",
        horizon=4,
        dt=1.0,
        controls=[ControlVariableSpec(name="u", low=0.0, high=1.0, initial=0.0)],
        states=[StateSpec(name="x", target=0.3, weight=1.0)],
    )


def test_offset_free_mpc_removes_steady_state_error_under_model_mismatch():
    kwargs = {
        "initial_state": np.array([0.0]),
        "surrogate": _biased_surrogate,
        "true_dynamics": _true_plant,
        "n_steps": 30,
    }
    plain = simulate_closed_loop(_mismatch_plan(), offset_free=False, **kwargs)
    corrected = simulate_closed_loop(_mismatch_plan(), **kwargs)
    # The biased model over-drives u; without correction the plant settles
    # around 0.4 instead of the 0.3 target.
    assert abs(plain["states"][-1, 0] - 0.3) > 0.05
    assert abs(corrected["states"][-1, 0] - 0.3) < 5e-3
    assert corrected["disturbance"].shape == (31, 1)


def test_offset_free_is_inert_when_model_matches_plant():
    kwargs = {
        "initial_state": np.array([0.0]),
        "surrogate": _true_plant,
        "true_dynamics": _true_plant,
        "n_steps": 10,
    }
    plain = simulate_closed_loop(_mismatch_plan(), offset_free=False, **kwargs)
    corrected = simulate_closed_loop(_mismatch_plan(), **kwargs)
    np.testing.assert_allclose(corrected["states"], plain["states"], atol=1e-9)
    np.testing.assert_allclose(corrected["disturbance"], 0.0, atol=1e-12)


def test_closed_loop_converges_on_linear_plant():
    plan = ControlPlan(
        name="thermostat",
        surrogate_uri="mem://scalar",
        horizon=5,
        dt=1.0,
        controls=[ControlVariableSpec(name="u", low=-1.0, high=1.0)],
        states=[StateSpec(name="x", target=0.0, weight=1.0)],
    )
    traj = simulate_closed_loop(
        plan,
        initial_state=np.array([2.0]),
        surrogate=_scalar_surrogate,
        true_dynamics=_scalar_surrogate,
        n_steps=10,
    )
    assert traj["states"].shape == (11, 1)
    assert abs(traj["states"][-1, 0]) < abs(traj["states"][0, 0])


def test_closed_loop_simulation_tool_returns_disturbance():
    from marimo_flow.agents.deps import FlowDeps
    from marimo_flow.agents.state import FlowState
    from marimo_flow.agents.toolsets.control import control_toolset

    class _Ctx:
        def __init__(self, deps):
            self.deps = deps

    deps = FlowDeps(state=FlowState(), provenance_db_path=":memory:")
    deps.registry["mem://biased"] = _biased_surrogate
    deps.registry["mem://plant"] = _true_plant
    out = control_toolset.tools["closed_loop_simulation"].function(
        _Ctx(deps),
        plan=_mismatch_plan().model_dump(),
        initial_state=[0.0],
        surrogate_registry_key="mem://biased",
        true_dynamics_registry_key="mem://plant",
        n_steps=30,
    )
    assert abs(out["states"][-1][0] - 0.3) < 5e-3
    assert len(out["disturbance"]) == 31
    assert out["disturbance"][-1][0] > 0.0
