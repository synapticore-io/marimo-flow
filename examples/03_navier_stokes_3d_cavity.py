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
trained with PINA's PINN solver. The lid profile is smooth (zero at its
edges); walls and lid are hard constraints of the network, the pressure
stays free. A finite-difference solver checks the result.

Training is cached on disk (``mo.persistent_cache``): the first run on a
CPU takes a while, re-opening the notebook is instant. For CI:
    marimo export html examples/03_navier_stokes_3d_cavity.py \
        -- --max-epochs 50 --lbfgs-epochs 2 --fd-grid 16
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

            {n_points} {width} {depth} {epochs} {lbfgs}
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
                value=int(mo.cli_args().get("max-epochs", 1500)),
                label="Adam steps",
            ),
            "lbfgs": mo.ui.slider(
                start=0,
                stop=200,
                step=1,
                value=int(mo.cli_args().get("lbfgs-epochs", 50)),
                label="L-BFGS rounds (20 iterations each)",
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

    # Regularised lid: the classic uniform lid jumps from U to 0 at the edges,
    # a singularity no smooth network can represent. This profile vanishes
    # on the lid edges and equals U in the middle.
    LID_PROFILE = "(16*x*(1 - x)*y*(1 - y))**2"
    lid_equations = [
        EquationSpec(
            name="lid_ux",
            form=f"ux - U*{LID_PROFILE}",
            outputs=["ux"],
            parameters={"U": float(cfg["lid_speed"])},
        ),
        EquationSpec(name="lid_uy", form="uy", outputs=["uy"]),
        EquationSpec(name="lid_uz", form="uz", outputs=["uz"]),
    ]

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
        + [continuity, *lid_equations],
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
                kind="equation",
                equation_names=["lid_ux", "lid_uy", "lid_uz"],
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
    from pina.optim import TorchOptimizer, TorchScheduler

    from marimo_flow.core import ModelManager, SolverManager, train_solver

    class HardWallCavity(torch.nn.Module):
        """``u = g + d·N``: walls and lid hold exactly, by construction.

        ``d`` vanishes on all six faces, ``g`` carries the lid profile and is
        zero on the other walls. The pressure ``p = N_p`` stays free.
        """

        def __init__(self, net: torch.nn.Module, lid_speed: float):
            super().__init__()
            self.net, self.lid_speed = net, lid_speed

        def forward(self, pts):
            x, y, z = (pts.extract([a]).as_subclass(torch.Tensor) for a in "xyz")
            raw = self.net(pts).as_subclass(torch.Tensor)
            bubble = 64 * x * (1 - x) * y * (1 - y) * z * (1 - z)
            lid = self.lid_speed * (16 * x * (1 - x) * y * (1 - y)) ** 2 * z
            return torch.cat(
                [lid + bubble * raw[:, :1], bubble * raw[:, 1:3], raw[:, 3:]], dim=1
            )

    def build_model(target_problem, width: int, depth: int, lid_speed: float):
        net = ModelManager.create(
            "feedforward", problem=target_problem, layers=[width] * depth
        )
        return HardWallCavity(net, lid_speed)

    def interior_only(full_spec):
        # The wall/lid conditions are satisfied exactly by the model, so only
        # the PDE condition contributes to the loss.
        return full_spec.model_copy(
            update={
                "conditions": [c for c in full_spec.conditions if c.subdomain == "D"]
            }
        )

    @mo.persistent_cache
    def train_cavity(
        spec_json: str,
        width: int,
        depth: int,
        lid_speed: float,
        adam_steps: int,
        lbfgs_rounds: int,
        points: int,
    ):
        torch.manual_seed(0)
        # Small tensors (points x width): beyond ~6 threads synchronisation
        # costs more than it saves — 16 threads ran 3x slower than 6.
        threads = torch.get_num_threads()
        torch.set_num_threads(min(6, threads))
        fresh = compose_problem(
            interior_only(ProblemSpec.model_validate_json(spec_json))
        )()
        model = build_model(fresh, width, depth, lid_speed)
        tracker = MetricTracker()
        start = time.perf_counter()
        # Stage 1: Adam, learning rate decaying 1e-3 -> 1e-4.
        adam = SolverManager.create(
            "pinn",
            problem=fresh,
            model=model,
            optimizer=TorchOptimizer(torch.optim.Adam, lr=1e-3),
            scheduler=TorchScheduler(
                torch.optim.lr_scheduler.ExponentialLR,
                gamma=0.1 ** (1 / max(adam_steps, 1)),
            ),
        )
        trainer = train_solver(
            adam,
            max_epochs=adam_steps,
            accelerator="cpu",
            n_points=points,
            sample_mode="latin",
            callbacks=[tracker],
        )
        history = tracker.metrics["train_loss"].detach().cpu().numpy().tolist()
        # Stage 2: L-BFGS polish on the same collocation points.
        if lbfgs_rounds:
            polish = SolverManager.create(
                "pinn",
                problem=fresh,
                model=model,
                optimizer=TorchOptimizer(
                    torch.optim.LBFGS,
                    lr=1.0,
                    max_iter=20,
                    history_size=50,
                    line_search_fn="strong_wolfe",
                ),
            )
            tracker = MetricTracker()
            trainer = train_solver(
                polish,
                max_epochs=lbfgs_rounds,
                accelerator="cpu",
                n_points=points,
                sample_mode="latin",
                callbacks=[tracker],
            )
            history += tracker.metrics["train_loss"].detach().cpu().numpy().tolist()
        seconds = time.perf_counter() - start
        torch.set_num_threads(threads)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        return state, metrics, history, seconds

    with mo.status.spinner(
        title=f"Training {cfg['epochs']} Adam steps + {cfg['lbfgs']} L-BFGS rounds ..."
    ):
        _state, metrics, loss_history, train_seconds = train_cavity(
            spec.model_dump_json(),
            int(cfg["width"]),
            int(cfg["depth"]),
            float(cfg["lid_speed"]),
            int(cfg["epochs"]),
            int(cfg["lbfgs"]),
            int(cfg["n_points"]),
        )
    model = build_model(problem, int(cfg["width"]), int(cfg["depth"]), cfg["lid_speed"])
    model.load_state_dict(_state)
    model.eval()
    solver = SolverManager.create("pinn", problem=problem, model=model)
    mo.vstack(
        [
            mo.md(
                f"## Training — {cfg['epochs']} Adam steps + {cfg['lbfgs']} "
                f"L-BFGS rounds, {train_seconds:.0f} s\n\n"
                "Walls and lid are hard constraints (`u = g + d·N`), so the loss "
                "is the PDE residual alone."
            ),
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
    # Zero up to float32 rounding: the walls are hard constraints.
    wall_slip = {"rms": float(np.sqrt((_speed**2).mean())), "max": float(_speed.max())}

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
                        f"{wall_slip['max']:.1e}", label="max |u| on no-slip walls"
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
def _validation(cfg, evaluate, go, mo, np, reynolds):
    from marimo_flow.core.cavity_reference import (
        cell_centres,
        centerlines,
        minimum,
        solve_cavity,
    )

    fd_grid = int(mo.cli_args().get("fd-grid", 32))

    @mo.persistent_cache
    def fd_reference(re: float, n: int, lid_speed: float):
        return solve_cavity(re, n, lid_speed=lid_speed, smooth=True)

    with mo.status.spinner(title=f"Finite-difference reference on {fd_grid}³ ..."):
        fd_vel = fd_reference(float(reynolds), fd_grid, float(cfg["lid_speed"]))
    pinn_vel = evaluate(cell_centres(fd_grid))[:, :3].reshape(fd_vel.shape)
    _s, _ux_fd, _uz_fd = centerlines(fd_vel)
    _, _ux_pinn, _uz_pinn = centerlines(pinn_vel)
    validation = {
        "rel_l2": float(np.linalg.norm(pinn_vel - fd_vel) / np.linalg.norm(fd_vel)),
        "max_abs": float(np.abs(pinn_vel - fd_vel).max()),
        "u_min_pinn": minimum(_s, _ux_pinn),
        "u_min_fd": minimum(_s, _ux_fd),
        "w_min_pinn": minimum(_s, _uz_pinn),
        "w_min_fd": minimum(_s, _uz_fd),
    }

    _fig = go.Figure()
    for _name, _ux, _uz, _dash in (
        ("FD", _ux_fd, _uz_fd, "solid"),
        ("PINN", _ux_pinn, _uz_pinn, "dash"),
    ):
        _fig.add_trace(
            go.Scatter(x=_ux, y=_s, name=f"u_x(z) {_name}", line={"dash": _dash})
        )
        _fig.add_trace(
            go.Scatter(x=_s, y=_uz, name=f"u_z(x) {_name}", line={"dash": _dash})
        )
    _fig.update_layout(
        title=f"Centerlines: PINN vs finite differences ({fd_grid}³)",
        xaxis_title="velocity / coordinate",
        yaxis_title="coordinate / velocity",
    )
    mo.vstack(
        [
            mo.md(
                "## Validation against a finite-difference reference\n\n"
                "Independent solver (`marimo_flow.core.cavity_reference`: MAC grid, "
                "Chorin projection), same smooth lid. It reproduces the published "
                "Re = 100 benchmark (Oueslati et al. 2011) within 1 %."
            ),
            mo.hstack(
                [
                    mo.stat(f"{validation['rel_l2']:.1%}", label="relative L2 error"),
                    mo.stat(f"{validation['max_abs']:.3f}", label="max |error|"),
                    mo.stat(
                        f"{validation['u_min_pinn'][0]:.4f}",
                        label=f"u_x,min PINN (FD {validation['u_min_fd'][0]:.4f})",
                    ),
                    mo.stat(
                        f"{validation['w_min_pinn'][0]:.4f}",
                        label=f"u_z,min PINN (FD {validation['w_min_fd'][0]:.4f})",
                    ),
                ],
                wrap=True,
            ),
            mo.ui.plotly(_fig),
        ]
    )
    return (validation,)


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
