"""Typed schemas for the PINA agent team (SPEC §8).

Composition-first: ``ProblemSpec`` is built from ``EquationSpec`` +
``SubdomainSpec`` + ``ConditionSpec`` primitives — no hardcoded kind
enum. Any PDE expressible in PINA is reachable without touching Python.
"""

from marimo_flow.agents.schemas.artifacts import (
    ArtifactKind,
    ArtifactRef,
    DatasetBinding,
)
from marimo_flow.agents.schemas.control import (
    ControlParameterSpec,
    ControlPlan,
    ControlVariableSpec,
    MPCSolver,
    StateSpec,
)
from marimo_flow.agents.schemas.decisions import (
    AgentDecision,
    AgentRole,
    ExperimentRecord,
    HandoffRecord,
    ValidationReport,
)
from marimo_flow.agents.schemas.equation import (
    ConditionKind,
    ConditionSpec,
    DerivativeKind,
    DerivativeSpec,
    EquationSpec,
    SubdomainSpec,
)
from marimo_flow.agents.schemas.mesh import CellKind, MeshFormat, MeshSpec
from marimo_flow.agents.schemas.observation import (
    ObservationSource,
    ObservationSpec,
    UnknownParameterSpec,
)
from marimo_flow.agents.schemas.optimization import (
    ConstraintOp,
    ConstraintSpec,
    DesignVariableSpec,
    OptimizationMethod,
    OptimizationPlan,
)
from marimo_flow.agents.schemas.preset import (
    PresetFamily,
    PresetRecord,
    PresetStatus,
)
from marimo_flow.agents.schemas.problem import ProblemSpec
from marimo_flow.agents.schemas.run import (
    ModelKind,
    ModelSpec,
    RunConfig,
    SolverKind,
    SolverPlan,
)
from marimo_flow.agents.schemas.stochastic import NoiseKind, NoiseSpec
from marimo_flow.agents.schemas.task import ProblemKindHint, TaskSpec

__all__ = [
    "AgentDecision",
    "AgentRole",
    "ArtifactKind",
    "ArtifactRef",
    "CellKind",
    "ConditionKind",
    "ConditionSpec",
    "ConstraintOp",
    "ConstraintSpec",
    "ControlParameterSpec",
    "ControlPlan",
    "ControlVariableSpec",
    "DatasetBinding",
    "DerivativeKind",
    "DerivativeSpec",
    "DesignVariableSpec",
    "EquationSpec",
    "ExperimentRecord",
    "HandoffRecord",
    "MeshFormat",
    "MeshSpec",
    "MPCSolver",
    "ModelKind",
    "ModelSpec",
    "NoiseKind",
    "NoiseSpec",
    "ObservationSource",
    "ObservationSpec",
    "OptimizationMethod",
    "OptimizationPlan",
    "PresetFamily",
    "PresetRecord",
    "PresetStatus",
    "ProblemKindHint",
    "ProblemSpec",
    "RunConfig",
    "SolverKind",
    "SolverPlan",
    "StateSpec",
    "SubdomainSpec",
    "TaskSpec",
    "UnknownParameterSpec",
    "ValidationReport",
]
