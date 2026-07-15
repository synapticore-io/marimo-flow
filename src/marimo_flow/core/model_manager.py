"""Model manager for creating PINA models via a single API.

Mirrors the Problem/Solver manager pattern: a registry of builder factories
keyed by a short `kind` string, a `create(kind, problem, **kwargs)` entry
point, and `register(kind, builder)` for project-local extensions.

Supported out of the box:
  * feedforward  - pina.model.FeedForward (plus residual variant)
  * residual     - pina.model.ResidualFeedForward
  * fno          - pina.model.FNO (Fourier Neural Operator)
  * deeponet     - pina.model.DeepONet
  * mionet       - pina.model.MIONet
  * pirate       - pina.model.PirateNet
  * walrus       - marimo_flow.core.walrus.FoundationModelAdapter (Hugging Face)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from pina.problem import BaseProblem


def _in_out_dims(problem: BaseProblem) -> tuple[int, int]:
    if problem.input_variables is None or problem.output_variables is None:
        raise ValueError("Problem must define input_variables and output_variables.")
    return len(problem.input_variables), len(problem.output_variables)


def _create_feedforward(
    problem: BaseProblem,
    *,
    layers: list[int] | None = None,
    activation: type[nn.Module] | None = None,
    **_ignored: Any,
) -> nn.Module:
    from pina.model import FeedForward

    in_dim, out_dim = _in_out_dims(problem)
    return FeedForward(
        input_dimensions=in_dim,
        output_dimensions=out_dim,
        layers=layers or [64, 64, 64],
        func=activation or nn.Tanh,  # type: ignore[arg-type]
    )


def _create_residual_feedforward(
    problem: BaseProblem,
    *,
    layers: list[int] | None = None,
    activation: type[nn.Module] | None = None,
    **_ignored: Any,
) -> nn.Module:
    from pina.model import ResidualFeedForward

    in_dim, out_dim = _in_out_dims(problem)
    return ResidualFeedForward(
        input_dimensions=in_dim,
        output_dimensions=out_dim,
        layers=layers or [64, 64, 64],
        func=activation or nn.Tanh,  # type: ignore[arg-type]
    )


def _create_fno(
    problem: BaseProblem,
    *,
    lifting_net: nn.Module | None = None,
    projecting_net: nn.Module | None = None,
    n_modes: int | list[int] = 8,
    dimensions: int = 1,
    inner_size: int = 32,
    n_layers: int = 4,
    activation: type[nn.Module] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Fourier Neural Operator.

    Requires `lifting_net` and `projecting_net` (either provided by the caller
    or built here from the problem's input/output dimensions as simple 1-layer
    nets, which is the common default in PINA tutorials).
    """
    from pina.model import FNO

    in_dim, out_dim = _in_out_dims(problem)
    lift = lifting_net or nn.Linear(in_dim, inner_size)
    proj = projecting_net or nn.Linear(inner_size, out_dim)
    return FNO(
        lifting_net=lift,
        projecting_net=proj,
        n_modes=n_modes,
        dimensions=dimensions,
        inner_size=inner_size,
        n_layers=n_layers,
        func=activation or nn.GELU,
        **kwargs,
    )


def _create_deeponet(
    problem: BaseProblem,
    *,
    branch_net: nn.Module | None = None,
    trunk_net: nn.Module | None = None,
    input_indices_branch_net: list[int] | None = None,
    input_indices_trunk_net: list[int] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """DeepONet — caller must supply `branch_net` + `trunk_net` (PINA has no
    sensible default for those without knowing the operator being learned)."""
    from pina.model import DeepONet

    if branch_net is None or trunk_net is None:
        raise ValueError(
            "DeepONet requires both `branch_net` and `trunk_net` to be provided."
        )
    return DeepONet(
        branch_net=branch_net,
        trunk_net=trunk_net,
        input_indices_branch_net=input_indices_branch_net or [0],
        input_indices_trunk_net=input_indices_trunk_net or [1],
        **kwargs,
    )


def _create_pirate(
    problem: BaseProblem,
    *,
    layers: list[int] | None = None,
    activation: type[nn.Module] | None = None,
    **kwargs: Any,
) -> nn.Module:
    from pina.model import PirateNet

    in_dim, out_dim = _in_out_dims(problem)
    return PirateNet(
        input_dimensions=in_dim,
        output_dimensions=out_dim,
        layers=layers or [64, 64, 64],
        func=activation or nn.Tanh,  # type: ignore[arg-type]
        **kwargs,
    )


def _create_walrus(
    problem: BaseProblem,
    *,
    checkpoint: str = "polymathic-ai/walrus",
    freeze_backbone: bool = True,
    **kwargs: Any,
) -> nn.Module:
    from marimo_flow.core.walrus import FoundationModelAdapter

    in_dim, out_dim = _in_out_dims(problem)
    out_labels = tuple(problem.output_variables or ("u",))
    return FoundationModelAdapter(
        checkpoint=checkpoint,
        input_dimensions=in_dim,
        out_labels=out_labels[:out_dim],
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


class ModelManager:
    """Single entry point for creating model instances from a problem."""

    _REGISTRY: dict[str, Callable[..., nn.Module]] = {
        "feedforward": _create_feedforward,
        "residual": _create_residual_feedforward,
        "fno": _create_fno,
        "deeponet": _create_deeponet,
        "pirate": _create_pirate,
        "walrus": _create_walrus,
    }

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._REGISTRY))

    @classmethod
    def register(cls, kind: str, builder: Callable[..., nn.Module]) -> None:
        cls._REGISTRY[kind.strip().lower()] = builder

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        problem: BaseProblem,
        **kwargs: Any,
    ) -> nn.Module:
        key = kind.strip().lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown model kind '{kind}'. Available: {', '.join(cls.available())}"
            )
        return cls._REGISTRY[key](problem=problem, **kwargs)
