"""ProblemSpec — compositional description of a PINA Problem.

No hardcoded ``kind`` enum. Agents construct a ProblemSpec from
primitives (equations, subdomains, conditions) and hand it to
``services.composer.compose_problem`` which assembles a live
``pina.Problem`` subclass at runtime. Any PDE that PINA's operators
can express is reachable without touching Python.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from marimo_flow.agents.schemas.control import ControlParameterSpec
from marimo_flow.agents.schemas.equation import (
    ConditionSpec,
    EquationSpec,
    SubdomainSpec,
)
from marimo_flow.agents.schemas.mesh import MeshSpec
from marimo_flow.agents.schemas.observation import (
    ObservationSpec,
    UnknownParameterSpec,
)
from marimo_flow.agents.schemas.stochastic import NoiseSpec


class ProblemSpec(BaseModel):
    """Compositional description of a PDE problem.

    The composer reads the fields as follows:

    * ``output_variables`` becomes the problem class attribute.
    * ``domain_bounds`` is the full ambient domain. Presence of ``"t"``
      promotes the problem to ``TimeDependentProblem`` (otherwise
      ``SpatialProblem``).
    * ``subdomains`` declares the named regions that appear in conditions
      (walls, initial-time slice, interior, …).
    * ``equations`` lists the PDE residuals; they are referenced by name
      from ``conditions[*].equation_name``.
    * ``conditions`` maps each subdomain to either an equation (interior
      PDE residual, IC expression) or a ``fixed_value`` (homogeneous
      Dirichlet etc.).
    * ``unknowns`` (optional) turns the problem into an inverse one —
      the composer adds ``pina.InverseProblem`` to the base classes and
      wires each unknown through ``params_`` in the residual.
    * ``observations`` (optional) are data-fitting conditions; each
      becomes a ``Condition(input=…, target=…)`` in the compiled problem.
    * ``control_parameters`` (optional) declare exogenous MPC inputs such
      as a boundary temperature ``u``; merged into ``domain_bounds`` and
      sampled during PINN training.

    Multiple entries in ``equations`` coexist without any extra schema —
    multiphysics coupling is just several EquationSpecs that share
    ``output_variables`` and point at the same interior subdomain.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        description="optional catalog label, e.g. 'burgers_1d_forward'",
    )
    output_variables: list[str]
    domain_bounds: dict[str, list[float]] = Field(
        description="full domain per axis, e.g. {'x': [-1.0, 1.0], 't': [0.0, 1.0]}"
    )
    subdomains: list[SubdomainSpec] = Field(default_factory=list)
    equations: list[EquationSpec] = Field(default_factory=list)
    conditions: list[ConditionSpec] = Field(default_factory=list)
    control_parameters: list[ControlParameterSpec] = Field(default_factory=list)
    unknowns: list[UnknownParameterSpec] = Field(default_factory=list)
    observations: list[ObservationSpec] = Field(default_factory=list)
    mesh: MeshSpec | None = Field(
        default=None,
        description="optional mesh; SubdomainSpec.mesh_ref references its tags",
    )
    noise: NoiseSpec | None = Field(
        default=None,
        description="optional additive stochastic term; MC-averaged residual",
    )
    notes: str | None = None

    @property
    def time_dependent(self) -> bool:
        """True iff ``domain_bounds`` includes a time axis."""
        return "t" in self.domain_bounds

    @property
    def is_inverse(self) -> bool:
        """True iff at least one unknown parameter is declared."""
        return bool(self.unknowns)
