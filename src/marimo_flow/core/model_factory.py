"""Thin backwards-compatible wrapper around ModelManager.

Kept so `examples/01_pina_poisson_solver.py` and other callers can use the
older `create_model_for_problem(...)` signature. New code should use
`ModelManager.create("feedforward"|"fno"|..., problem=p, **kwargs)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from pina.model import FeedForward

from marimo_flow.core.model_manager import ModelManager

if TYPE_CHECKING:
    from pina.problem import BaseProblem


def create_model_for_problem(
    problem: BaseProblem,
    *,
    layers: list[int] | None = None,
    activation: type[nn.Module] | None = None,
) -> FeedForward:
    """Create one feedforward model sized from a problem definition."""
    return ModelManager.create(  # type: ignore[return-value]
        "feedforward",
        problem=problem,
        layers=layers,
        activation=activation,
    )
