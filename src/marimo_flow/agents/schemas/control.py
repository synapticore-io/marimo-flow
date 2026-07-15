"""ControlPlan — declarative recipe for a PINN-surrogate MPC loop.

Scope:

* Rolling-horizon model predictive control with a trained PINN as the
  dynamics surrogate.
* Scalar control variables, scalar / vector state, quadratic or
  arbitrary sympy objective.
* scipy.optimize SLSQP as the inner QP solver — keeps the dep surface
  small. For bigger / convex problems escalate to cvxpy (soft-dep).

Out of scope:

* Real-time closed-loop hardware interfaces (MQTT / ROS bridges).
* Hybrid / mixed-integer MPC — needs a different solver.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MPCSolver = Literal["scipy_slsqp", "cvxpy"]


class ControlParameterSpec(BaseModel):
    """Exogenous parameter wired into a PINA problem (e.g. boundary temperature).

    Declared on ``ProblemSpec.control_parameters`` and merged into
    ``domain_bounds`` by the composer so collocation samples ``u`` and the
    network sees it as an input coordinate.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    low: float
    high: float


class ControlVariableSpec(BaseModel):
    """One actuator / control input the MPC may set at each step."""

    model_config = ConfigDict(extra="forbid")

    name: str
    low: float
    high: float
    initial: float = 0.0


class StateSpec(BaseModel):
    """One state channel read from the surrogate / observation model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    target: float | None = Field(
        default=None,
        description="setpoint used by the default quadratic objective",
    )
    weight: float = Field(default=1.0, description="weight in the quadratic cost")


class ControlPlan(BaseModel):
    """Recipe for a closed-loop MPC run driven by a PINN surrogate.

    ``surrogate_uri`` is a reference into ``deps.registry`` (same
    mechanism as ``ArtifactRef.uri``) — callers hand the trained
    solver/problem pair keyed by URI. The control toolset pulls the
    surrogate, calls ``solver.forward`` at each predicted horizon step.

    ``objective_expression`` is optional; when absent a quadratic cost
    over ``(state - target)`` is used instead with the declared weights.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    surrogate_uri: str
    horizon: int = Field(ge=1, description="MPC prediction horizon (steps)")
    dt: float = Field(gt=0.0, description="physical time-step between MPC calls")
    controls: list[ControlVariableSpec] = Field(default_factory=list)
    states: list[StateSpec] = Field(default_factory=list)
    objective_expression: str | None = None
    constraints: list[str] = Field(
        default_factory=list,
        description="sympy inequalities over state + control, e.g. 'T <= 400'",
    )
    solver: MPCSolver = "scipy_slsqp"
    notes: str | None = None
