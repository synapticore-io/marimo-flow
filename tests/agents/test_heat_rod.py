"""Tests for the 1D heat-rod plant + reduced-order surrogate."""

from __future__ import annotations

import numpy as np

from marimo_flow.agents.schemas import (
    ControlPlan,
    ControlVariableSpec,
    StateSpec,
)
from marimo_flow.control import (
    FiniteDifferenceHeatRod,
    generate_step_dataset,
    make_plant_stepper,
    make_rollout_surrogate,
    run_mpc_step,
    simulate_closed_loop,
    train_step_surrogate,
)


def test_rollout_surrogate_chains_state():
    def predict(t_centre: float, u: float) -> float:
        return 0.9 * t_centre + 0.1 * u

    surrogate = make_rollout_surrogate(predict)
    traj = surrogate(np.array([1.0]), np.array([[0.5], [0.5]]))
    assert traj.shape == (2, 1)
    assert traj[0, 0] == 0.9 * 1.0 + 0.1 * 0.5
    assert traj[1, 0] == 0.9 * traj[0, 0] + 0.1 * 0.5


def test_plant_stepper_advances_field():
    plant = FiniteDifferenceHeatRod(nx=11, alpha=0.05, dt=0.01)
    plant.set_uniform(0.0)
    stepper = make_plant_stepper(plant)
    traj = stepper(np.array([0.0]), np.full((10, 1), 1.0))
    assert traj.shape == (10, 1)
    assert traj[-1, 0] > 0.0


def test_trained_surrogate_enables_mpc_step():
    rng = np.random.default_rng(0)
    plant = FiniteDifferenceHeatRod(nx=21, alpha=0.05, dt=0.01)
    inputs, targets = generate_step_dataset(plant, 400, rng)
    predict = train_step_surrogate(inputs, targets, epochs=30)
    surrogate = make_rollout_surrogate(predict)

    plan = ControlPlan(
        name="heat_rod",
        surrogate_uri="mem://heat",
        horizon=4,
        dt=0.01,
        controls=[ControlVariableSpec(name="u", low=0.0, high=1.0, initial=0.5)],
        states=[StateSpec(name="T_mean", target=0.6, weight=1.0)],
    )
    ctrl, info = run_mpc_step(plan, np.array([0.0]), surrogate)
    assert ctrl.shape == (4, 1)
    assert np.all(ctrl >= -1e-9)
    assert np.all(ctrl <= 1.0 + 1e-9)
    assert info["success"]


def test_closed_loop_plant_moves_toward_setpoint():
    rng = np.random.default_rng(1)
    plant = FiniteDifferenceHeatRod(nx=11, alpha=0.05, dt=0.01)
    inputs, targets = generate_step_dataset(plant, 800, rng)
    predict = train_step_surrogate(inputs, targets, epochs=60)
    surrogate = make_rollout_surrogate(predict)

    loop_plant = FiniteDifferenceHeatRod(nx=11, alpha=0.05, dt=0.01)
    loop_plant.set_uniform(0.0)
    plant_step = make_plant_stepper(loop_plant)

    plan = ControlPlan(
        name="heat_rod",
        surrogate_uri="mem://heat",
        horizon=6,
        dt=0.01,
        controls=[ControlVariableSpec(name="u", low=0.0, high=1.0, initial=0.5)],
        states=[StateSpec(name="T_mean", target=0.6, weight=1.0)],
    )
    traj = simulate_closed_loop(
        plan,
        initial_state=np.array([0.0]),
        surrogate=surrogate,
        true_dynamics=plant_step,
        n_steps=40,
    )
    assert traj["states"][-1, 0] > traj["states"][0, 0]
    assert np.mean(traj["controls"][:, 0]) > 0.1
