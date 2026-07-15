"""PINN field model → scalar MPC surrogate (reduced-order mean temperature)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from pina.label_tensor import LabelTensor

from marimo_flow.control.heat_rod import SurrogateFn

ObserveKind = Literal["mean", "center"]


def query_field_mean(
    solver: Any,
    *,
    t: float,
    u: float,
    output_field: str = "T",
    nx: int = 21,
    input_order: list[str] | None = None,
) -> float:
    """Evaluate a trained PINN on a spatial line and return the spatial mean."""
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    labels = input_order or ["x", "t", "u"]
    columns: list[np.ndarray] = []
    for label in labels:
        if label == "x":
            columns.append(xs)
        elif label == "t":
            columns.append(np.full(nx, t, dtype=np.float64))
        elif label == "u":
            columns.append(np.full(nx, u, dtype=np.float64))
        else:
            raise ValueError(f"unsupported input label {label!r} in query")
    pts = LabelTensor(
        torch.tensor(np.column_stack(columns), dtype=torch.float32),
        labels,
    )
    values = _solver_forward(solver, pts, output_field)
    return float(np.mean(values))


def make_pinn_rollout_surrogate(
    solver: Any,
    *,
    dt: float,
    output_field: str = "T",
    nx: int = 21,
    input_order: list[str] | None = None,
    blend: float = 0.6,
) -> SurrogateFn:
    """Chain reduced-order PINN predictions for MPC horizon planning.

    At each horizon step the PINN field mean at ``(t=(k+1)*dt, u_k)`` is
    blended with the rolled scalar state so the optimizer sees state
    dependence while the field comes from the trained network.
    """

    def surrogate(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        horizon = len(controls)
        traj = np.zeros((horizon, state.shape[0]), dtype=np.float64)
        x = float(state[0])
        for k in range(horizon):
            u = float(controls[k, 0])
            t = (k + 1) * dt
            field_mean = query_field_mean(
                solver,
                t=t,
                u=u,
                output_field=output_field,
                nx=nx,
                input_order=input_order,
            )
            x = (1.0 - blend) * x + blend * field_mean
            traj[k, 0] = x
        return traj

    return surrogate


def register_pinn_surrogate(
    registry: dict[str, Any],
    key: str,
    solver: Any,
    *,
    dt: float,
    output_field: str = "T",
    nx: int = 21,
    blend: float = 0.6,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Store a PINN rollout surrogate in ``deps.registry`` and return it."""
    fn = make_pinn_rollout_surrogate(
        solver,
        dt=dt,
        output_field=output_field,
        nx=nx,
        blend=blend,
    )
    registry[key] = fn
    return fn


def _solver_forward(solver: Any, pts: LabelTensor, output_field: str) -> np.ndarray:
    solver.eval()
    with torch.no_grad():
        out = solver(pts)
    if hasattr(out, "extract"):
        tensor = out.extract([output_field]).detach().cpu().numpy()
    elif hasattr(out, "detach"):
        tensor = out.detach().cpu().numpy()
    else:
        tensor = np.asarray(out)
    return np.asarray(tensor, dtype=np.float64).reshape(-1)
