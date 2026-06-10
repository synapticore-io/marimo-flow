"""Training helpers for the PINA demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import mlflow
import torch
from pina.label_tensor import LabelTensor
from pina.solver import PINN, SupervisedSolver
from pina.trainer import Trainer

# torch >= 2.6 flips torch.load's default to weights_only=True. PINA checkpoints
# embed LabelTensor instances, so mlflow.pytorch autolog's model reload fails to
# unpickle them unless the class is allowlisted. Registering it is idempotent.
torch.serialization.add_safe_globals([LabelTensor])

_UNSET = object()
_FALLBACK_LIGHTNING_ROOT = Path("data") / "mlflow" / "lightning"


def _resolve_default_root_dir() -> Path:
    """Pick a stable directory for Lightning's logs + checkpoints.

    Lightning's ``Trainer`` defaults ``default_root_dir`` to
    ``os.getcwd()`` and writes ``checkpoints/`` and ``lightning_logs/``
    relative to that. For marimo-flow that pollutes the repo root or
    whatever directory the notebook was launched from. Pin it instead:

    1. If an MLflow run is active and uses a local file-store backend,
       point Lightning at the run's artifact directory — the ``.ckpt``
       files then land *inside* the run and show up in MLflow's
       artifact tab.
    2. Otherwise fall back to ``data/mlflow/lightning/`` next to the
       MLflow backend store, so we never escape into the working dir.
    """
    active = mlflow.active_run()
    if active is not None:
        parsed = urlparse(active.info.artifact_uri)
        if parsed.scheme in ("", "file"):
            local = unquote(parsed.path)
            # urlparse on Windows file:///C:/foo yields '/C:/foo' — strip
            # the leading slash so Path() doesn't treat it as POSIX-rooted.
            if (
                sys.platform == "win32"
                and local.startswith("/")
                and len(local) > 2
                and local[2] == ":"
            ):
                local = local[1:]
            return Path(local)
    return _FALLBACK_LIGHTNING_ROOT


def train_solver(
    solver: PINN | SupervisedSolver,
    max_epochs: int = 1000,
    accelerator: str = "auto",
    callbacks: list[Any] | None = None,
    logger: Any = _UNSET,
    n_points: int = 1000,
    sample_mode: str = "random",
    default_root_dir: str | os.PathLike[str] | None = None,
) -> Trainer:
    """Train the provided solver and return the fitted Trainer.

    ``logger`` defaults to ``False`` (no logger). Pass an explicit Lightning
    logger (CSVLogger, MLflowLogger, …) to enable per-step metrics. The
    Lightning default ``logger=True`` would emit a UserWarning about
    ``tensorboardX`` being absent — opt out of that auto-default here.

    ``default_root_dir`` defaults to the active MLflow run's artifact
    directory (local file-stores) or ``data/mlflow/lightning/`` otherwise,
    so Lightning's ``checkpoints/`` and ``lightning_logs/`` never escape
    into the working directory. Pass an explicit path to override.
    """
    solver.problem.discretise_domain(n=n_points, mode=sample_mode, domains="all")
    root_dir = (
        Path(default_root_dir)
        if default_root_dir is not None
        else _resolve_default_root_dir()
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        solver=solver,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks or [],
        logger=False if logger is _UNSET else logger,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=str(root_dir),
    )
    trainer.train()
    return trainer
