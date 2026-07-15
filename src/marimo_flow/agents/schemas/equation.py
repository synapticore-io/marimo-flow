"""EquationSpec + SubdomainSpec + ConditionSpec — composition primitives.

The Problem agent produces these instead of picking a hardcoded kind.
The composer (``services.composer.compose_problem``) turns them into a
live ``pina.Problem`` subclass at runtime.

Typed + explicit so agents can reason about PDEs structurally (which
field, which derivative, which parameter) rather than writing Python.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DerivativeKind = Literal["classical", "fractional_laplacian"]


class DerivativeSpec(BaseModel):
    """One symbolic derivative used in an equation.

    Example for ``u_t`` (∂u/∂t): ``name="u_t"``, ``field="u"``, ``wrt=["t"]``.
    For ``u_xx`` (∂²u/∂x²): ``wrt=["x", "x"]``. For mixed ``u_xy``:
    ``wrt=["x", "y"]``.

    ``kind="fractional_laplacian"`` switches to a spectral / Riesz-kernel
    quadrature with order ``alpha`` (0 < α < 2). ``wrt`` then lists the
    spatial axes the fractional Laplacian integrates over (usually all
    spatial coords).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="label used in the equation form, e.g. 'u_t'")
    field: str = Field(description="output variable being differentiated, e.g. 'u'")
    wrt: list[str] = Field(
        description="input vars to differentiate against in order, e.g. ['x','x']"
    )
    kind: DerivativeKind = "classical"
    alpha: float | None = Field(
        default=None,
        description="fractional order α in (0, 2); required for fractional_laplacian",
    )
    quadrature_points: int = Field(
        default=64,
        description="kernel quadrature resolution for fractional derivatives",
    )


class EquationSpec(BaseModel):
    """A symbolic PDE residual (the part set to zero).

    ``form`` is a sympy-parseable expression using the labels declared in
    ``derivatives`` (e.g. ``u_t``, ``u_xx``), the output variable names
    (``u``, ``v``, ``p``), the input variable names (``x``, ``y``, ``z``,
    ``t``), and any names in ``parameters``. The composer compiles the
    form into a torch callable via ``sympy.lambdify`` and wires the
    derivatives through ``pina.operator.grad`` / ``laplacian``.

    Example for 1D viscous Burgers (u_t + u·u_x − ν·u_xx = 0)::

        EquationSpec(
            name="burgers",
            form="u_t + u*u_x - nu*u_xx",
            outputs=["u"],
            derivatives=[
                DerivativeSpec(name="u_t",  field="u", wrt=["t"]),
                DerivativeSpec(name="u_x",  field="u", wrt=["x"]),
                DerivativeSpec(name="u_xx", field="u", wrt=["x","x"]),
            ],
            parameters={"nu": 0.01},
        )
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    form: str = Field(description="sympy expression; residual form (= 0)")
    outputs: list[str] = Field(
        description="output vars referenced in the form, e.g. ['u']"
    )
    derivatives: list[DerivativeSpec] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)


class SubdomainSpec(BaseModel):
    """A named sub-region of the full domain.

    Two mutually exclusive ways to declare the region:

    * ``bounds`` — axis → range dict. A single-element list or a scalar
      pins the coordinate (boundary wall / initial-time slice); a
      2-element list is an interval. 1D time-dependent interior ``D``
      on x∈[-1,1], t∈[0,1]::

          SubdomainSpec(name="D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]})

      Left wall (x=-1, t∈[0,1])::

          SubdomainSpec(name="left", bounds={"x": -1.0, "t": [0.0, 1.0]})

    * ``mesh_ref`` — human tag name declared under
      ``MeshSpec.cell_tags`` / ``point_tags``. The composer resolves
      the tag to a subset of the mesh and wraps it in a MeshDomain.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    bounds: dict[str, float | list[float]] = Field(default_factory=dict)
    mesh_ref: str | None = Field(
        default=None,
        description="tag name declared on the attached MeshSpec",
    )


ConditionKind = Literal["fixed_value", "equation", "parametric_dirichlet"]


class ConditionSpec(BaseModel):
    """Attach an equation (or a fixed scalar) to a named subdomain.

    ``kind="fixed_value"`` + ``value=0.0`` → homogeneous Dirichlet.
    ``kind="equation"`` + ``equation_name="..."`` → points at an
    EquationSpec defined at the ProblemSpec level (either the main PDE
    residual applied on an interior subdomain, or a BC/IC expression).
    ``kind="parametric_dirichlet"`` + ``parameter_name="u"`` → enforces
    ``output_field = u`` on the subdomain (``u`` sampled as an input).
    ``equation_inline`` is an escape hatch for one-off conditions
    (e.g. a custom IC) that shouldn't clutter the top-level equations list.
    """

    model_config = ConfigDict(extra="forbid")

    subdomain: str
    kind: ConditionKind
    value: float | None = None
    equation_name: str | None = None
    equation_inline: EquationSpec | None = None
    parameter_name: str | None = Field(
        default=None,
        description="control-parameter axis for parametric_dirichlet",
    )
    output_field: str | None = Field(
        default=None,
        description="field pinned to parameter_name; defaults to first output",
    )
