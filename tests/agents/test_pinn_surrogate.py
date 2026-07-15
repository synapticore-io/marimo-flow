"""Tests for PINN → MPC surrogate adapter."""

from __future__ import annotations

import numpy as np
import torch
from pina.label_tensor import LabelTensor

from marimo_flow.control.pinn_surrogate import (
    make_pinn_rollout_surrogate,
    query_field_mean,
)


class _MockSolver(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pts: LabelTensor) -> LabelTensor:
        u = pts.extract(["u"]).tensor
        return LabelTensor(u * 0.5 + 0.1, ["T"])


def test_query_field_mean_reads_pinn_output():
    solver = _MockSolver()
    mean = query_field_mean(solver, t=0.1, u=0.8, output_field="T", nx=5)
    assert abs(mean - 0.5) < 1e-5


def test_pinn_rollout_surrogate_returns_horizon_shape():
    solver = _MockSolver()
    surrogate = make_pinn_rollout_surrogate(solver, dt=0.01, blend=1.0)
    traj = surrogate(np.array([0.0]), np.array([[0.2], [0.4], [0.6]]))
    assert traj.shape == (3, 1)
    assert traj[0, 0] > 0.0
