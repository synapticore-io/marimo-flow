# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # Pulls pina-mathlab, torch, plotly, ... Declared explicitly so marimo's
#     # sandbox does not auto-install the unrelated PyPI package "pina".
#     "marimo-flow>=0.4.5",
# ]
# ///

"""3D lid-driven cavity: steady incompressible Navier-Stokes via the composer.

Run with:
    marimo edit examples/03_navier_stokes_3d_cavity.py

The full system — three momentum equations plus continuity — is declared
as a ProblemSpec, compiled at runtime by the sympy-backed composer and
trained with PINA's PINN solver. Walls are no-slip on the velocity only;
the pressure stays free.

Training is cached on disk (``mo.persistent_cache``): the first run on a
CPU takes a while, re-opening the notebook is instant. For CI:
    marimo export html examples/03_navier_stokes_3d_cavity.py -- --max-epochs 50
"""

from __future__ import annotations

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _header():
    import marimo as mo

    mo.md(
        "# Lid-driven cavity — 3D Navier–Stokes\n"
        "\n"
        "Steady, incompressible flow in the unit cube. The lid at `z = 1` "
        "moves in `x`; all other walls are no-slip. Unknowns: `ux, uy, uz, p`.\n"
        "\n"
        "$$(\\mathbf{u}\\cdot\\nabla)\\mathbf{u} + \\nabla p = "
        "\\nu\\,\\nabla^2\\mathbf{u}, \\qquad \\nabla\\cdot\\mathbf{u} = 0$$\n"
        "\n"
        "Composed from primitives — no hardcoded PDE factory. The notebook "
        "trains once with the defaults on load; change the settings and "
        "press **Train** to run again."
    )
    return (mo,)


@app.cell
def _controls(mo):
    cavity_form = mo.ui.batch(
        mo.md(
            """
            {viscosity} {lid_speed}

            {n_points} {width} {depth} {epochs}
            """
        ),
        {
            "viscosity": mo.ui.slider(
                start=0.01, stop=1.0, step=0.01, value=0.1, label="ν (viscosity)"
            ),
            "lid_speed": mo.ui.slider(
                start=0.1, stop=2.0, step=0.1, value=1.0, label="lid speed"
            ),
            "n_points": mo.ui.slider(
                start=500, stop=8000, step=500, value=2000, label="points per region"
            ),
            "width": mo.ui.slider(
                start=32, stop=128, step=16, value=64, label="hidden width"
            ),
            "depth": mo.ui.slider(start=2, stop=6, step=1, value=4, label="layers"),
            "epochs": mo.ui.slider(
                start=10,
                stop=5000,
                step=10,
                value=int(mo.cli_args().get("max-epochs", 3000)),
                label="epochs (full-batch steps)",
            ),
        },
    ).form(submit_button_label="Train", bordered=True)
    cavity_form
    return (cavity_form,)


@app.cell
def _cfg(cavity_form, mo):
    cfg = cavity_form.value or cavity_form.element.value
    reynolds = cfg["lid_speed"] / cfg["viscosity"]
    mo.md(f"Reynolds number `Re = U·L/ν = {reynolds:.0f}`")
    return cfg, reynolds


@app.cell
def _build_spec(cfg):
    from marimo_flow.agents.schemas import (
        ConditionSpec,
        DerivativeSpec,
        EquationSpec,
        ProblemSpec,
        SubdomainSpec,
    )

    AXES = ("x", "y", "z")
    VELOCITY = ("ux", "uy", "uz")
    BOX = {a: [0.0, 1.0] for a in AXES}
    WALLS = {
        "floor": ("z", 0.0),
        "xlow": ("x", 0.0),
        "xhigh": ("x", 1.0),
        "ylow": ("y", 0.0),
        "yhigh": ("y", 1.0),
    }

    def momentum(comp: str, axis: str, nu: float) -> EquationSpec:
        derivs = [DerivativeSpec(name=f"{comp}_{a}", field=comp, wrt=[a]) for a in AXES]
        derivs += [
            DerivativeSpec(name=f"{comp}_{a}{a}", field=comp, wrt=[a, a]) for a in AXES
        ]
        derivs.append(DerivativeSpec(name=f"p_{axis}", field="p", wrt=[axis]))
        convection = " + ".join(
            f"{v}*{comp}_{a}" for v, a in zip(VELOCITY, AXES, strict=True)
        )
        diffusion = " + ".join(f"{comp}_{a}{a}" for a in AXES)
        return EquationSpec(
            name=f"mom_{axis}",
            form=f"{convection} + p_{axis} - nu*({diffusion})",
            outputs=[*VELOCITY, "p"],
            derivatives=derivs,
            parameters={"nu": nu},
        )

    continuity = EquationSpec(
        name="continuity",
        form="ux_x + uy_y + uz_z",
        outputs=list(VELOCITY),
        derivatives=[
            DerivativeSpec(name=f"{v}_{a}", field=v, wrt=[a])
            for v, a in zip(VELOCITY, AXES, strict=True)
        ],
    )

    spec = ProblemSpec(
        name=f"ns3d_cavity_nu{cfg['viscosity']:.2f}",
        output_variables=[*VELOCITY, "p"],
        domain_bounds=BOX,
        subdomains=[
            SubdomainSpec(name="D", bounds=BOX),
            SubdomainSpec(name="lid", bounds={**BOX, "z": 1.0}),
        ]
        + [SubdomainSpec(name=n, bounds={**BOX, a: v}) for n, (a, v) in WALLS.items()],
        equations=[
            momentum(c, a, float(cfg["viscosity"]))
            for c, a in zip(VELOCITY, AXES, strict=True)
        ]
        + [continuity],
        conditions=[
            # Momentum (x, y, z) + continuity, all enforced in the interior.
            ConditionSpec(
                subdomain="D",
                kind="equation",
                equation_names=["mom_x", "mom_y", "mom_z", "continuity"],
            ),
            # Velocity-only Dirichlet: the pressure is not pinned on walls.
            ConditionSpec(
                subdomain="lid",
                kind="fixed_value",
                values={"ux": float(cfg["lid_speed"]), "uy": 0.0, "uz": 0.0},
            ),
        ]
        + [
            ConditionSpec(
                subdomain=n, kind="fixed_value", values=dict.fromkeys(VELOCITY, 0.0)
            )
            for n in WALLS
        ],
    )
    return ProblemSpec, spec


@app.cell
def _compose(spec):
    from marimo_flow.agents.services.composer import compose_problem

    problem = compose_problem(spec)()
    return compose_problem, problem


@app.cell
def _show_domain(mo, problem):
    from marimo_flow.core.viz3d import domain_figure

    mo.vstack([mo.md("## Spatial domain"), mo.ui.plotly(domain_figure(problem))])
    return


@app.cell
def _train(ProblemSpec, cfg, compose_problem, mo, problem, spec):
    import time

    import torch
    from pina.callback import MetricTracker

    from marimo_flow.core import ModelManager, SolverManager, train_solver

    def build_solver(target_problem, width: int, depth: int):
        model = ModelManager.create(
            "feedforward", problem=target_problem, layers=[width] * depth
        )
        solver = SolverManager.create(
            "pinn", problem=target_problem, model=model, learning_rate=1e-3
        )
        return model, solver

    @mo.persistent_cache
    def train_cavity(spec_json: str, width: int, depth: int, epochs: int, points: int):
        torch.manual_seed(0)
        fresh = compose_problem(ProblemSpec.model_validate_json(spec_json))()
        model, solver = build_solver(fresh, width, depth)
        tracker = MetricTracker()
        start = time.perf_counter()
        trainer = train_solver(
            solver,
            max_epochs=epochs,
            accelerator="cpu",
            n_points=points,
            sample_mode="latin",
            callbacks=[tracker],
        )
        seconds = time.perf_counter() - start
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        history = tracker.metrics["train_loss"].detach().cpu().numpy().tolist()
        return state, metrics, history, seconds

    with mo.status.spinner(title=f"Training {cfg['epochs']} steps on CPU ..."):
        _state, metrics, loss_history, train_seconds = train_cavity(
            spec.model_dump_json(),
            int(cfg["width"]),
            int(cfg["depth"]),
            int(cfg["epochs"]),
            int(cfg["n_points"]),
        )
    model, solver = build_solver(problem, int(cfg["width"]), int(cfg["depth"]))
    model.load_state_dict(_state)
    model.eval()
    mo.vstack(
        [
            mo.md(f"## Training — {cfg['epochs']} steps, {train_seconds:.0f} s"),
            mo.ui.table([metrics], selection=None),
        ]
    )
    return loss_history, metrics, solver, torch, train_seconds


@app.cell
def _diagnostics(mo, problem, solver, torch):
    import numpy as np
    from pina.label_tensor import LabelTensor

    def evaluate(points: np.ndarray) -> np.ndarray:
        inp = LabelTensor(torch.tensor(points, dtype=torch.float32), ["x", "y", "z"])
        with torch.no_grad():
            return np.asarray(solver.forward(inp).tensor)

    # PDE residuals of the trained network at fresh interior points, computed
    # with the very equations the composer built (mom_x, mom_y, mom_z, cont).
    _rng = np.random.default_rng(1)
    _interior = LabelTensor(
        torch.tensor(_rng.random((4000, 3)), dtype=torch.float32), ["x", "y", "z"]
    )
    _interior.requires_grad_(True)
    _res = problem.conditions["D"].equation.residual(
        _interior, solver.forward(_interior)
    )
    rms = np.sqrt((np.asarray(_res.detach().tensor) ** 2).mean(axis=0))
    residual_rms = dict(
        zip(["mom_x", "mom_y", "mom_z", "continuity"], rms.tolist(), strict=True)
    )

    # No-slip check on random wall points (floor + four side walls).
    _w = _rng.random((5000, 3))
    _axis = _rng.integers(0, 3, 5000)
    _side = _rng.integers(0, 2, 5000).astype(float)
    _w[np.arange(5000), _axis] = _side
    _w = _w[~((_axis == 2) & (_side == 1.0))]  # drop lid points
    _speed = np.linalg.norm(evaluate(_w)[:, :3], axis=1)
    wall_slip = {
        "rms": float(np.sqrt((_speed**2).mean())),
        # The maximum sits on the lid edges, where the velocity jumps from
        # the lid speed to zero (corner singularity of the cavity problem).
        "max": float(_speed.max()),
    }

    mo.vstack(
        [
            mo.md("## Physics diagnostics (fresh points, not training points)"),
            mo.hstack(
                [
                    mo.stat(f"{v:.1e}", label=f"RMS residual {k}")
                    for k, v in residual_rms.items()
                ]
                + [
                    mo.stat(
                        f"{wall_slip['rms']:.3f}", label="RMS |u| on no-slip walls"
                    ),
                    mo.stat(
                        f"{wall_slip['max']:.3f}",
                        label="max |u| on walls (lid edges)",
                    ),
                ],
                wrap=True,
            ),
        ]
    )
    return evaluate, np, residual_rms, wall_slip


@app.cell
def _centerline(cfg, evaluate, mo, np):
    import plotly.graph_objects as go

    _s = np.linspace(0.0, 1.0, 101)
    _half = np.full_like(_s, 0.5)
    ux_centerline = evaluate(np.stack([_half, _half, _s], 1))[:, 0]
    uz_centerline = evaluate(np.stack([_s, _half, _half], 1))[:, 2]
    _i = int(ux_centerline.argmin())
    return_flow = {"ux_min": float(ux_centerline[_i]), "z_at_min": float(_s[_i])}

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=ux_centerline, y=_s, name="u_x(z) at x = y = 0.5"))
    _fig.add_trace(go.Scatter(x=_s, y=uz_centerline, name="u_z(x) at y = z = 0.5"))
    _fig.update_layout(
        title=(
            f"Centerline profiles — return flow u_x,min = {return_flow['ux_min']:.3f} "
            f"at z = {return_flow['z_at_min']:.2f} (lid speed {cfg['lid_speed']})"
        ),
        xaxis_title="velocity / coordinate",
        yaxis_title="coordinate / velocity",
    )
    mo.ui.plotly(_fig)
    return return_flow, ux_centerline


@app.cell
def _slice(evaluate, go, mo, np):
    _n = 41
    _s = np.linspace(0.0, 1.0, _n)
    _X, _Z = np.meshgrid(_s, _s)
    _out = evaluate(np.stack([_X.ravel(), np.full(_X.size, 0.5), _Z.ravel()], 1))
    slice_ux = _out[:, 0].reshape(_n, _n)
    slice_uz = _out[:, 2].reshape(_n, _n)
    _speed = np.hypot(slice_ux, slice_uz)

    _step = 4
    _fig = go.Figure(go.Heatmap(x=_s, y=_s, z=_speed, colorbar={"title": "|u|"}))
    _fig.add_trace(
        go.Scatter(
            x=_X[::_step, ::_step].ravel(),
            y=_Z[::_step, ::_step].ravel(),
            mode="markers",
            marker={
                "symbol": "arrow",
                "size": 10,
                "angle": np.degrees(
                    np.arctan2(slice_ux, slice_uz)[::_step, ::_step]
                ).ravel(),
                "color": "white",
            },
            showlegend=False,
        )
    )
    _fig.update_layout(
        title="Mid-plane y = 0.5: speed and flow direction (x, z)",
        xaxis_title="x",
        yaxis_title="z",
        height=520,
    )
    mo.ui.plotly(_fig)
    return slice_ux, slice_uz


@app.cell
def _loss(go, loss_history, mo):
    _fig = go.Figure(go.Scatter(y=loss_history, mode="lines"))
    _fig.update_layout(
        title="Training loss", yaxis_type="log", xaxis_title="step", yaxis_title="loss"
    )
    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
