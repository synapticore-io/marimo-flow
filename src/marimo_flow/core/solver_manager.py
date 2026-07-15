"""Solver manager for creating PINA solvers via a single API.

Registers the common PINN-family solvers (PINN, SAPINN, CausalPINN,
GradientPINN, RBAPINN) and the SupervisedSolver. Advanced solvers that
need auxiliary nets (CompetitivePINN, GAROM) are left out of the default
registry; register them per-project via `SolverManager.register(...)`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from pina.optim import TorchOptimizer
from pina.problem import BaseProblem

# PINA 0.3 renamed the solver classes to verbose *SingleModelSolver forms;
# alias back to the established short domain names (PINN, SAPINN, …).
from pina.solver import (
    CausalPhysicsInformedSingleModelSolver as CausalPINN,
)
from pina.solver import (
    GradientPhysicsInformedSingleModelSolver as GradientPINN,
)
from pina.solver import (
    PhysicsInformedSingleModelSolver as PINN,
)
from pina.solver import (
    RBAPhysicsInformedSingleModelSolver as RBAPINN,
)
from pina.solver import (
    SelfAdaptivePhysicsInformedSolver as SAPINN,
)
from pina.solver import (
    SupervisedSingleModelSolver as SupervisedSolver,
)


def _resolve_optimizer(
    *,
    optimizer: torch.optim.Optimizer | TorchOptimizer | None,
    learning_rate: float,
    optimizer_type: type[torch.optim.Optimizer] | None,
) -> tuple[torch.optim.Optimizer | TorchOptimizer, type[torch.optim.Optimizer]]:
    resolved_type = optimizer_type or torch.optim.Adam
    if optimizer is not None:
        return optimizer, resolved_type
    return TorchOptimizer(resolved_type, lr=learning_rate), resolved_type


def _make_standard_pinn_factory(
    solver_cls: type,
) -> Callable[..., Any]:
    """Factory builder for PINN-family solvers sharing the standard signature
    `(problem, model, optimizer, ...)`. Covers PINN, CausalPINN, GradientPINN,
    RBAPINN.
    """

    def _factory(
        problem: BaseProblem,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
        learning_rate: float = 1e-3,
        optimizer_type: type[torch.optim.Optimizer] | None = None,
        **solver_kwargs: Any,
    ) -> Any:
        resolved_optimizer, _ = _resolve_optimizer(
            optimizer=optimizer,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
        )
        return solver_cls(
            problem=problem,
            model=model,
            optimizer=resolved_optimizer,
            **solver_kwargs,
        )

    return _factory


def _create_supervised_solver(
    problem: BaseProblem,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
    learning_rate: float = 1e-3,
    optimizer_type: type[torch.optim.Optimizer] | None = None,
    loss: nn.Module | None = None,
    use_lt: bool = False,
    **solver_kwargs: Any,
) -> SupervisedSolver:
    resolved_optimizer, _ = _resolve_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
    )
    return SupervisedSolver(
        problem=problem,
        model=model,
        optimizer=resolved_optimizer,
        loss=loss or nn.MSELoss(),
        use_lt=use_lt,
        **solver_kwargs,
    )


def _create_sapinn(
    problem: BaseProblem,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
    learning_rate: float = 1e-3,
    optimizer_type: type[torch.optim.Optimizer] | None = None,
    **solver_kwargs: Any,
) -> SAPINN:
    optimizer_model, resolved_type = _resolve_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
    )
    return SAPINN(
        problem=problem,
        model=model,
        optimizer_model=optimizer_model,
        optimizer_weights=TorchOptimizer(resolved_type, lr=learning_rate),
        **solver_kwargs,
    )


class SolverManager:
    """Single entry point for creating solver instances."""

    _REGISTRY: dict[str, Callable[..., Any]] = {
        "pinn": _make_standard_pinn_factory(PINN),
        "causalpinn": _make_standard_pinn_factory(CausalPINN),
        "gradientpinn": _make_standard_pinn_factory(GradientPINN),
        "rbapinn": _make_standard_pinn_factory(RBAPINN),
        "sapinn": _create_sapinn,
        "supervised": _create_supervised_solver,
    }

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._REGISTRY))

    @classmethod
    def register(cls, kind: str, builder: Callable[..., Any]) -> None:
        cls._REGISTRY[kind.strip().lower()] = builder

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        problem: BaseProblem,
        model: nn.Module,
        **kwargs: Any,
    ) -> Any:
        key = kind.strip().lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown solver kind '{kind}'. Available: {', '.join(cls.available())}"
            )
        return cls._REGISTRY[key](problem=problem, model=model, **kwargs)
