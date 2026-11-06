"""Smoke-tests that the Phase B-F demo notebooks still compose.

Re-imports the `ProblemSpec`/`ControlPlan` values used by the demo
notebooks and runs the composer + one forward pass. Guards against
schema drift breaking the user-facing notebooks without the unit
tests noticing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from pina import LabelTensor

from marimo_flow.agents.schemas import (
    ControlPlan,
    ControlVariableSpec,
    StateSpec,
)
from marimo_flow.agents.services.composer import compose_problem
from marimo_flow.control import make_rollout_surrogate, run_mpc_step


def _load_notebook(name: str):
    path = Path(__file__).resolve().parents[2] / "examples" / name
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_ns3d_cavity_demo_composes_full_navier_stokes():
    """Runs the notebook's own spec cell, so the demo cannot drift."""
    notebook = _load_notebook("03_navier_stokes_3d_cavity.py")
    _, defs = notebook._build_spec.run(cfg={"viscosity": 0.1, "lid_speed": 1.0})
    problem = compose_problem(defs["spec"])()
    assert problem.input_variables == ["x", "y", "z"]
    assert problem.output_variables == ["ux", "uy", "uz", "p"]
    walls = {"lid", "floor", "xlow", "xhigh", "ylow", "yhigh"}
    assert set(problem.conditions) == {"D"} | walls

    pts = LabelTensor(torch.rand(16, 3), ["x", "y", "z"])
    pts.requires_grad_(True)
    x, y, z = (pts.extract([a]) for a in ("x", "y", "z"))
    out = LabelTensor(
        torch.cat([torch.sin(x) * y, x * z, y * z, x + y + z], dim=1),
        ["ux", "uy", "uz", "p"],
    )
    # Momentum x/y/z + continuity in the interior.
    assert problem.conditions["D"].equation.residual(pts, out).shape == (16, 4)
    # Lid pins the three velocity components to (1, 0, 0), not the pressure.
    lid = problem.conditions["lid"].equation.residual(pts, out)
    expected = out.extract(["ux", "uy", "uz"]) - torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(lid, expected)


def test_heat_rod_demo_composes_and_mpc_step_runs():
    from marimo_flow.control.heat_rod import build_heat_rod_problem_spec

    problem = compose_problem(build_heat_rod_problem_spec(alpha=0.08))()
    assert set(problem.input_variables or []) == {"x", "t", "u"}

    def predict(t_centre: float, u: float) -> float:
        return 0.85 * t_centre + 0.15 * u

    surrogate = make_rollout_surrogate(predict)

    plan = ControlPlan(
        name="heat_rod_mpc_demo",
        surrogate_uri="mem://demo",
        horizon=3,
        dt=0.05,
        controls=[ControlVariableSpec(name="u", low=-1.0, high=1.0)],
        states=[StateSpec(name="T_centre", target=0.5, weight=1.0)],
    )
    u_seq, info = run_mpc_step(plan, state_now=np.array([0.0]), surrogate=surrogate)
    assert u_seq.shape == (3, 1)
    assert np.all(np.abs(u_seq) <= 1.0 + 1e-9)
    assert info["success"]
