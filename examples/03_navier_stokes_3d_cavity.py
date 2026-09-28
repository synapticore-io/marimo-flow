# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # Pulls pina-mathlab, torch, plotly, ... Declared explicitly so marimo's
#     # sandbox does not auto-install the unrelated PyPI package "pina".
#     "marimo-flow>=0.4.3",
# ]
# ///

"""3D lid-driven cavity via the PINA composer.

Run with:
    marimo edit examples/03_navier_stokes_3d_cavity.py

Demonstrates the composition-first architecture end-to-end on a
three-dimensional problem: no hardcoded ProblemKind, no built-in
NS factory — just a declarative ProblemSpec, the sympy-backed
composer, and PINA's FeedForward + PINN solver.

Keeps viscosity high and epochs low so the notebook runs in a few
minutes on CPU. For real NS research bump n_points to 40k+, hidden
width to 128+, and switch to LBFGS for the last stretch.
"""

from __future__ import annotations

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _header():
    import marimo as mo

    mo.md(
        "# Lid-driven cavity — 3D Navier–Stokes\n"
        "Composed from primitives, no hardcoded PDE factory. Change a "
        "viscosity, add an obstacle subdomain, whatever — re-run and the "
        "composer rebuilds the PINA class at runtime."
    )
    return (mo,)


@app.cell
def _controls(mo):
    viscosity = mo.ui.slider(
        start=0.01, stop=1.0, step=0.01, value=0.1, label="ν (viscosity)"
    )
    lid_speed = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=1.0, label="lid speed"
    )
    n_points = mo.ui.slider(
        start=2000, stop=20000, step=2000, value=8000, label="collocation points"
    )
    hidden_width = mo.ui.slider(
        start=32, stop=128, step=16, value=64, label="hidden width"
    )
    max_epochs = mo.ui.slider(start=5, stop=100, step=5, value=20, label="max epochs")
    mo.vstack(
        [
            mo.hstack([viscosity, lid_speed]),
            mo.hstack([n_points, hidden_width, max_epochs]),
        ]
    )
    return (viscosity, lid_speed, n_points, hidden_width, max_epochs)


@app.cell
def _build_spec(viscosity, lid_speed):
    from marimo_flow.agents.schemas import (
        ConditionSpec,
        DerivativeSpec,
        EquationSpec,
        ProblemSpec,
        SubdomainSpec,
    )

    BOX = {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}

    spec = ProblemSpec(
        name=f"ns3d_cavity_nu{viscosity.value:.2f}",
        output_variables=["ux", "uy", "uz", "p"],
        domain_bounds=BOX,
        subdomains=[
            SubdomainSpec(name="D", bounds=BOX),
            SubdomainSpec(name="lid", bounds={**BOX, "z": 1.0}),
            SubdomainSpec(name="floor", bounds={**BOX, "z": 0.0}),
            SubdomainSpec(name="xlow", bounds={**BOX, "x": 0.0}),
            SubdomainSpec(name="xhigh", bounds={**BOX, "x": 1.0}),
            SubdomainSpec(name="ylow", bounds={**BOX, "y": 0.0}),
            SubdomainSpec(name="yhigh", bounds={**BOX, "y": 1.0}),
        ],
        equations=[
            # Momentum in each direction (steady, incompressible).
            EquationSpec(
                name="mom_x",
                form="ux*ux_x + uy*ux_y + uz*ux_z + p_x - nu*(ux_xx + ux_yy + ux_zz)",
                outputs=["ux", "uy", "uz", "p"],
                derivatives=[
                    DerivativeSpec(name="ux_x", field="ux", wrt=["x"]),
                    DerivativeSpec(name="ux_y", field="ux", wrt=["y"]),
                    DerivativeSpec(name="ux_z", field="ux", wrt=["z"]),
                    DerivativeSpec(name="ux_xx", field="ux", wrt=["x", "x"]),
                    DerivativeSpec(name="ux_yy", field="ux", wrt=["y", "y"]),
                    DerivativeSpec(name="ux_zz", field="ux", wrt=["z", "z"]),
                    DerivativeSpec(name="p_x", field="p", wrt=["x"]),
                ],
                parameters={"nu": float(viscosity.value)},
            ),
            EquationSpec(
                name="continuity",
                form="ux_x + uy_y + uz_z",
                outputs=["ux", "uy", "uz"],
                derivatives=[
                    DerivativeSpec(name="ux_x", field="ux", wrt=["x"]),
                    DerivativeSpec(name="uy_y", field="uy", wrt=["y"]),
                    DerivativeSpec(name="uz_z", field="uz", wrt=["z"]),
                ],
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="mom_x"),
            # Continuity lives on a second interior copy; see pina-multiphysics.
            # For this demo we attach it inline to a distinct face to keep a
            # single condition per subdomain name.
            ConditionSpec(
                subdomain="xhigh", kind="equation", equation_name="continuity"
            ),
            # Walls: u = 0 everywhere except the lid where ux = lid_speed.
            ConditionSpec(subdomain="floor", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="xlow", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="ylow", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="yhigh", kind="fixed_value", value=0.0),
            ConditionSpec(
                subdomain="lid", kind="fixed_value", value=float(lid_speed.value)
            ),
        ],
    )
    return (spec,)


@app.cell
def _compose(spec):
    from marimo_flow.agents.services.composer import compose_problem

    cls = compose_problem(spec)
    problem = cls()
    return (problem,)


@app.cell
def _show_domain(mo, problem):
    from marimo_flow.core.viz3d import domain_figure

    domain_fig = domain_figure(problem)
    mo.vstack([mo.md("## Spatial domain"), mo.ui.plotly(domain_fig)])
    return (domain_fig,)


@app.cell
def _train(mo, problem, n_points, hidden_width, max_epochs):
    from marimo_flow.core import ModelManager, SolverManager, train_solver

    model = ModelManager.create(
        "feedforward",
        problem=problem,
        layers=[hidden_width.value, hidden_width.value, hidden_width.value],
    )
    solver = SolverManager.create(
        "pinn", problem=problem, model=model, learning_rate=1e-3
    )
    trainer = train_solver(
        solver,
        max_epochs=max_epochs.value,
        accelerator="cpu",
        n_points=n_points.value,
        sample_mode="latin",
    )
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    mo.vstack([mo.md("## Training metrics"), mo.ui.table([metrics])])
    return (trainer, solver, metrics)


@app.cell
def _predict_slice(mo, solver):
    import numpy as np
    import torch
    from pina.label_tensor import LabelTensor

    grid = 24
    xs = np.linspace(0, 1, grid)
    ys = np.linspace(0, 1, grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, 0.5)  # mid-plane slice
    coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    inp = LabelTensor(torch.tensor(coords, dtype=torch.float32), ["x", "y", "z"])
    out = solver.forward(inp).detach().cpu().numpy()
    ux_slice = out[:, 0].reshape(grid, grid)

    import plotly.graph_objects as go

    slice_fig = go.Figure(data=go.Heatmap(x=xs, y=ys, z=ux_slice, colorscale="Viridis"))
    slice_fig.update_layout(title="u_x at z = 0.5 after training")
    mo.ui.plotly(slice_fig)
    return (slice_fig, ux_slice)


if __name__ == "__main__":
    app.run()
