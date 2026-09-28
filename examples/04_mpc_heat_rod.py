# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # Pulls pina-mathlab, torch, plotly, ... Declared explicitly so marimo's
#     # sandbox does not auto-install the unrelated PyPI package "pina".
#     "marimo-flow>=0.4.3",
# ]
# ///

"""Closed-loop MPC on a 1D heat rod with a PINN surrogate.

Run with:
    marimo edit examples/04_mpc_heat_rod.py

Trains a parametric PINN (``T(x,t,u)`` with ``T(1,t,u)=u``) via the
composer, wraps it as a reduced-order MPC surrogate, and closes the loop
against an explicit finite-difference plant.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    ALPHA = 0.5


@app.cell
def _header():
    import marimo as mo

    mo.md(
        "# MPC demo — heat rod with PINN surrogate\n"
        "\n"
        "1. **ProblemSpec** — 1D heat equation, ``T(1)=u`` (parametric BC)\n"
        "2. **PINN** — `compose_problem` → `train_solver`\n"
        "3. **Surrogate** — PINN field mean → MPC rollout\n"
        "4. **Plant** — finite-difference ground truth in the closed loop\n"
        "\n"
        "The rod is held at 0 on the left and at `u` ≤ 1 on the right, so the "
        "steady mean temperature is `u / 2`: setpoints above 0.5 are "
        "physically unreachable. The surrogate is biased; offset-free MPC "
        "estimates that bias as a disturbance `d` and removes the "
        "steady-state error."
    )
    return (mo,)


@app.cell
def _controls(mo):
    setpoint = mo.ui.slider(
        start=0.0, stop=0.5, step=0.05, value=0.3, label="mean-T setpoint"
    )
    horizon = mo.ui.slider(start=3, stop=20, step=1, value=8, label="MPC horizon")
    n_steps = mo.ui.slider(
        start=10, stop=100, step=10, value=60, label="closed-loop steps"
    )
    max_epochs = mo.ui.slider(
        start=20, stop=200, step=10, value=80, label="PINN epochs"
    )
    n_points = mo.ui.slider(
        start=1000, stop=8000, step=1000, value=3000, label="collocation points"
    )
    mo.vstack(
        [
            mo.hstack([setpoint, horizon, n_steps]),
            mo.hstack([max_epochs, n_points]),
        ]
    )
    return horizon, max_epochs, mo, n_points, n_steps, setpoint


@app.cell
def _compose():
    from marimo_flow.agents.services.composer import compose_problem
    from marimo_flow.control.heat_rod import build_heat_rod_problem_spec

    # alpha = 0.5 gives a slowest time constant of 1/(alpha*pi^2) ~ 0.2, so
    # the rod settles within the PINN's t in [0, 1] and 60 steps of dt=0.01.
    spec = build_heat_rod_problem_spec(alpha=ALPHA)
    problem = compose_problem(spec)()
    return problem, spec


@app.cell
def _train_pinn(mo, max_epochs, n_points, problem):
    from marimo_flow.core import ModelManager, SolverManager, train_solver

    model = ModelManager.create("feedforward", problem=problem, layers=[32, 32])
    solver = SolverManager.create(
        "pinn", problem=problem, model=model, learning_rate=1e-3
    )
    trainer = train_solver(
        solver,
        max_epochs=int(max_epochs.value),
        accelerator="cpu",
        n_points=int(n_points.value),
        sample_mode="random",
    )
    mo.md(
        f"PINN trained for **{max_epochs.value}** epochs "
        f"({n_points.value} collocation points). "
        "Final losses: "
        + ", ".join(
            f"`{k}` {float(v):.4f}" for k, v in trainer.callback_metrics.items()
        )
    )
    return solver, trainer


@app.cell
def _wire_dynamics(solver):
    from marimo_flow.control.heat_rod import FiniteDifferenceHeatRod, make_plant_stepper
    from marimo_flow.control.pinn_surrogate import make_pinn_rollout_surrogate

    # nx=11 keeps the explicit FTCS scheme stable: alpha*dt/dx^2 = 0.5.
    plant = FiniteDifferenceHeatRod(nx=11, alpha=ALPHA, dt=0.01)
    plant.set_uniform(0.0)
    dt = 0.01
    surrogate = make_pinn_rollout_surrogate(
        solver, dt=dt, output_field="T", nx=21, blend=0.6
    )
    true_dynamics = make_plant_stepper(plant)
    return dt, plant, surrogate, true_dynamics


@app.cell
def _run_mpc(dt, horizon, mo, n_steps, setpoint, surrogate, true_dynamics):
    import numpy as np

    from marimo_flow.agents.schemas import (
        ControlPlan,
        ControlVariableSpec,
        StateSpec,
    )
    from marimo_flow.control import simulate_closed_loop

    plan = ControlPlan(
        name="heat_rod_mpc",
        surrogate_uri="local://heat_rod_pinn_surrogate",
        horizon=int(horizon.value),
        dt=dt,
        controls=[ControlVariableSpec(name="u", low=0.0, high=1.0, initial=0.5)],
        states=[StateSpec(name="T_mean", target=float(setpoint.value), weight=1.0)],
    )
    traj = simulate_closed_loop(
        plan,
        initial_state=np.array([0.0]),
        surrogate=surrogate,
        true_dynamics=true_dynamics,
        n_steps=int(n_steps.value),
    )

    import plotly.graph_objects as go

    states = traj["states"][:, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=states, mode="lines+markers", name="T_mean (plant)"))
    fig.add_trace(
        go.Scatter(
            y=[setpoint.value] * len(states),
            mode="lines",
            line={"dash": "dash"},
            name="setpoint",
        )
    )
    fig.update_layout(title="Closed-loop spatial mean (FD plant)", yaxis_title="T_mean")

    ctrl_fig = go.Figure()
    ctrl_fig.add_trace(
        go.Scatter(
            y=traj["controls"][:, 0],
            mode="lines+markers",
            name="u (right BC)",
        )
    )
    ctrl_fig.update_layout(title="Applied boundary temperature", yaxis_title="u")

    dist_fig = go.Figure()
    dist_fig.add_trace(
        go.Scatter(y=traj["disturbance"][:, 0], mode="lines", name="d (estimate)")
    )
    dist_fig.update_layout(
        title="Surrogate bias estimated by the offset-free observer",
        yaxis_title="d",
    )

    final_err = abs(states[-1] - setpoint.value)
    mo.vstack(
        [
            mo.md(f"Final tracking error |T_mean − setpoint| = **{final_err:.4f}**."),
            mo.ui.plotly(fig),
            mo.ui.plotly(ctrl_fig),
            mo.ui.plotly(dist_fig),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
