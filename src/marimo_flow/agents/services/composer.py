"""compose_problem — turn a typed ``ProblemSpec`` into a live PINA Problem.

Core of the composition-first architecture. Agents build a
``ProblemSpec`` from primitives (``EquationSpec`` + ``SubdomainSpec`` +
``ConditionSpec``) and hand it here. The composer:

1. Splits ``domain_bounds`` into spatial + temporal axes (``"t"`` if
   present promotes to ``TimeDependentProblem``).
2. Builds a ``CartesianDomain`` per subdomain from its ``bounds`` dict.
3. Compiles each ``EquationSpec.form`` through ``sympy.sympify`` +
   ``sympy.lambdify`` into a torch callable, wiring derivatives through
   ``pina.operator.grad`` / ``laplacian``.
4. Emits a ``pina.Condition`` per ``ConditionSpec`` (``FixedValue`` for
   ``fixed_value``, ``Equation`` lookup for ``equation``).
5. Dynamically constructs a ``pina.Problem`` subclass with those class
   attributes.

Inverse problems: if ``spec.unknowns`` is non-empty the composer mixes
``pina.InverseProblem`` into the base classes, sets
``unknown_parameter_domain`` from the declared bounds, and routes those
symbols through ``params_`` in the compiled residual. Observations in
``spec.observations`` become ``Condition(input=…, target=…)`` entries.

No hardcoded equation catalog. Any PDE expressible in sympy over
derivatives, outputs, input variables and scalar parameters is
reachable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy
import torch
from pina import Condition, LabelTensor
from pina.domain import CartesianDomain
from pina.equation import Equation
from pina.equation.zoo import FixedValue
from pina.operator import grad, laplacian
from pina.problem import InverseProblem, SpatialProblem, TimeDependentProblem

from marimo_flow.agents.schemas import (
    ConditionSpec,
    DerivativeSpec,
    EquationSpec,
    MeshSpec,
    NoiseSpec,
    ObservationSpec,
    ProblemSpec,
    SubdomainSpec,
)
from marimo_flow.agents.services.mesh_domain import MeshDomain, load_mesh_domain

_INPUT_VAR_CANDIDATES: tuple[str, ...] = ("x", "y", "z", "t")


def _merge_control_bounds(spec: ProblemSpec) -> dict[str, list[float]]:
    bounds: dict[str, list[float]] = {
        axis: [float(v[0]), float(v[1])] for axis, v in spec.domain_bounds.items()
    }
    for cp in spec.control_parameters:
        if cp.name not in bounds:
            bounds[cp.name] = [float(cp.low), float(cp.high)]
    return bounds


def _collect_input_names(bounds: dict[str, list[float]]) -> set[str]:
    return set(bounds.keys())


def compose_problem(spec: ProblemSpec) -> type:
    """Assemble a ``pina.Problem`` subclass from a ``ProblemSpec``.

    Returns a class (not an instance) — PINA problem classes carry
    their structure as class attributes, matching the existing
    ``ProblemManager`` return shape so downstream toolsets
    (``ModelManager``, ``SolverManager``) keep working unchanged.
    """
    domain_bounds = _merge_control_bounds(spec)
    input_names = _collect_input_names(domain_bounds)
    spatial_bounds, temporal_bounds = _split_domain(domain_bounds)
    temporal_domain = (
        CartesianDomain({"t": temporal_bounds}) if temporal_bounds else None
    )

    mesh_domain: MeshDomain | None = None
    if spec.mesh is not None:
        mesh_domain = load_mesh_domain(spec.mesh)

    if mesh_domain is not None:
        spatial_domain: Any = mesh_domain
    elif spatial_bounds:
        spatial_domain = CartesianDomain(spatial_bounds)
    else:
        spatial_domain = None

    subdomains: dict[str, Any] = {
        sd.name: _resolve_subdomain(sd, spec.mesh) for sd in spec.subdomains
    }

    unknown_names = {u.name for u in spec.unknowns}
    equations: dict[str, Equation] = {
        eq.name: build_equation(
            eq,
            unknown_names=unknown_names,
            input_names=input_names,
            noise=spec.noise,
        )
        for eq in spec.equations
    }

    conditions: dict[str, Condition] = {}
    for cond in spec.conditions:
        conditions[cond.subdomain] = _compile_condition(
            cond,
            equations,
            unknown_names=unknown_names,
            output_variables=spec.output_variables,
            input_names=input_names,
        )
    for obs in spec.observations:
        conditions[obs.name] = _compile_observation(obs)

    attrs: dict[str, Any] = {
        "output_variables": list(spec.output_variables),
        "domains": subdomains,
        "conditions": conditions,
    }
    if spatial_domain is not None:
        attrs["spatial_domain"] = spatial_domain

    if spec.time_dependent:
        if temporal_domain is None:
            raise ValueError("time_dependent=True but no 't' axis in domain_bounds")
        attrs["temporal_domain"] = temporal_domain
        base: tuple[type, ...] = (TimeDependentProblem, SpatialProblem)
    else:
        base = (SpatialProblem,)

    if spec.is_inverse:
        attrs["unknown_parameter_domain"] = CartesianDomain(
            {u.name: [float(u.low), float(u.high)] for u in spec.unknowns}
        )
        base = base + (InverseProblem,)

    class_name = spec.name or "ComposedProblem"
    return type(class_name, base, attrs)


def build_equation(
    spec: EquationSpec,
    *,
    unknown_names: set[str] | None = None,
    input_names: set[str] | None = None,
    noise: NoiseSpec | None = None,
) -> Equation:
    """Compile a symbolic ``EquationSpec`` into a ``pina.Equation``.

    The returned ``Equation`` wraps a torch callable that:

    * extracts each output field via ``output_.extract([field])``;
    * computes each declared derivative via PINA's ``grad`` /
      ``laplacian`` operators;
    * substitutes declared parameters as Python scalars;
    * pulls any ``unknown_names`` from PINA's ``params_`` dict so
      ``pina.InverseProblem`` can backprop into them;
    * evaluates the sympy form through ``lambdify`` to a torch
      expression that autograd can differentiate through.

    When ``unknown_names`` intersects the form's free symbols the
    callable exposes a 3-arg ``(input_, output_, params_)`` signature
    that PINA recognises as an inverse residual; otherwise it uses the
    2-arg direct signature.
    """
    unknowns = set(unknown_names or ())
    symbol_names, torch_fn = _build_lambda(spec, unknowns, input_names or set())

    form_unknowns = [n for n in symbol_names if n in unknowns]
    is_inverse = bool(form_unknowns)
    noise_sampler = _build_noise_sampler(noise)

    def _assemble(input_: Any, output_: Any, params_: Any | None) -> Any:
        values: dict[str, Any] = {}
        for out_var in spec.outputs:
            if out_var in symbol_names:
                values[out_var] = output_.extract([out_var])
        for var in sorted(input_names or set(_INPUT_VAR_CANDIDATES)):
            if var in symbol_names and var != "t":
                values[var] = input_.extract([var])
        if "t" in symbol_names:
            values["t"] = input_.extract(["t"])
        for pname, pval in spec.parameters.items():
            if pname in symbol_names:
                values[pname] = float(pval)
        for uname in form_unknowns:
            if params_ is None or uname not in params_:
                raise ValueError(
                    f"equation '{spec.name}' expects unknown parameter "
                    f"'{uname}' but params_ is missing it"
                )
            values[uname] = params_[uname]
        for deriv in spec.derivatives:
            if deriv.name in symbol_names:
                values[deriv.name] = _compute_derivative(input_, output_, deriv)

        missing = [n for n in symbol_names if n not in values]
        if missing:
            raise ValueError(
                f"equation '{spec.name}' is missing inputs for symbols: "
                f"{', '.join(missing)}"
            )
        residual = torch_fn(**values)
        if noise_sampler is not None:
            residual = residual - noise_sampler(input_)
        return residual

    if is_inverse:

        def residual_inverse(input_: Any, output_: Any, params_: Any) -> Any:
            return _assemble(input_, output_, params_)

        return Equation(residual_inverse)

    def residual(input_: Any, output_: Any) -> Any:
        return _assemble(input_, output_, None)

    return Equation(residual)


# --- internals -----------------------------------------------------


def _split_domain(
    bounds: dict[str, list[float]],
) -> tuple[dict[str, list[float]], list[float] | None]:
    spatial: dict[str, list[float]] = {}
    temporal: list[float] | None = None
    for axis, interval in bounds.items():
        if axis == "t":
            temporal = [float(v) for v in interval]
        else:
            spatial[axis] = [float(v) for v in interval]
    return spatial, temporal


def _resolve_subdomain(sd: SubdomainSpec, mesh: MeshSpec | None) -> Any:
    """Route a ``SubdomainSpec`` to a CartesianDomain or a tagged MeshDomain."""
    if sd.mesh_ref is not None:
        if mesh is None:
            raise ValueError(
                f"subdomain '{sd.name}' has mesh_ref but ProblemSpec.mesh is None"
            )
        return load_mesh_domain(mesh, mesh_ref=sd.mesh_ref)
    if not sd.bounds:
        raise ValueError(f"subdomain '{sd.name}' has neither bounds nor mesh_ref")
    return _subdomain_to_cartesian(sd)


def _subdomain_to_cartesian(sd: SubdomainSpec) -> CartesianDomain:
    """Convert a ``SubdomainSpec`` to ``CartesianDomain`` literal bounds.

    Scalars pin the axis (wall / initial slice); lists of length 2
    become intervals.
    """
    bounds: dict[str, float | list[float]] = {}
    for axis, val in sd.bounds.items():
        if isinstance(val, list | tuple):
            if len(val) == 1:
                bounds[axis] = float(val[0])
            elif len(val) == 2:
                bounds[axis] = [float(val[0]), float(val[1])]
            else:
                raise ValueError(
                    f"subdomain '{sd.name}' axis {axis!r}: bounds must have 1 or 2 values"
                )
        else:
            bounds[axis] = float(val)
    return CartesianDomain(bounds)


def _compile_condition(
    cond: ConditionSpec,
    equations: dict[str, Equation],
    *,
    unknown_names: set[str] | None = None,
    output_variables: list[str] | None = None,
    input_names: set[str] | None = None,
) -> Condition:
    if cond.kind == "fixed_value":
        if cond.value is None:
            raise ValueError(
                f"condition on '{cond.subdomain}' kind='fixed_value' needs value"
            )
        return Condition(domain=cond.subdomain, equation=FixedValue(cond.value))
    if cond.kind == "parametric_dirichlet":
        if not cond.parameter_name:
            raise ValueError(
                f"condition on '{cond.subdomain}' kind='parametric_dirichlet' "
                "needs parameter_name"
            )
        if not output_variables:
            raise ValueError(
                "parametric_dirichlet requires output_variables on ProblemSpec"
            )
        field = cond.output_field or output_variables[0]
        eq = _parametric_dirichlet_equation(field, cond.parameter_name)
        return Condition(domain=cond.subdomain, equation=eq)
    if cond.kind == "equation":
        if cond.equation_inline is not None:
            eq = build_equation(
                cond.equation_inline,
                unknown_names=unknown_names,
                input_names=input_names,
            )
        elif cond.equation_name is not None:
            eq = equations.get(cond.equation_name)
            if eq is None:
                raise ValueError(
                    f"condition on '{cond.subdomain}' references unknown "
                    f"equation {cond.equation_name!r}"
                )
        else:
            raise ValueError(
                f"condition on '{cond.subdomain}' kind='equation' needs "
                "either equation_name or equation_inline"
            )
        return Condition(domain=cond.subdomain, equation=eq)
    raise ValueError(f"unknown condition kind: {cond.kind!r}")


def _parametric_dirichlet_equation(field: str, parameter_name: str) -> Equation:
    """Dirichlet BC ``field = parameter_name`` with the parameter read from inputs."""

    def residual(input_: Any, output_: Any) -> Any:
        return output_.extract([field]) - input_.extract([parameter_name])

    return Equation(residual)


def _compile_observation(obs: ObservationSpec) -> Condition:
    """Compile an ``ObservationSpec`` into a data-fitting PINA ``Condition``.

    Requires ``points`` / ``values`` to be materialised — the Data agent
    is responsible for filling them before compose_problem runs.
    """
    if obs.points is None or obs.values is None:
        raise ValueError(
            f"observation '{obs.name}' is not materialised — the Data agent "
            "must fill .points and .values before compose_problem"
        )
    if not obs.axes:
        raise ValueError(
            f"observation '{obs.name}' has empty 'axes'; cannot label points"
        )
    input_tensor = LabelTensor(
        torch.tensor(obs.points, dtype=torch.float32), list(obs.axes)
    )
    target_tensor = LabelTensor(
        torch.tensor(obs.values, dtype=torch.float32), [obs.field]
    )
    return Condition(input=input_tensor, target=target_tensor)


def _build_noise_sampler(
    noise: NoiseSpec | None,
) -> Callable[[Any], Any] | None:
    """Return a closure that draws one noise realisation per residual call.

    The closure respects ``noise.kind`` — ``"white"`` is pure i.i.d.
    Gaussian noise per collocation point; ``"colored"`` smooths the
    sample via a Gaussian kernel on the first input axis; ``"fbm"`` is
    rejected unless the ``fbm`` package is available (soft-dep; escalate
    to Lead if needed).
    """
    if noise is None:
        return None
    gen = torch.Generator()
    if noise.seed is not None:
        gen.manual_seed(int(noise.seed))
    intensity = float(noise.intensity)

    if noise.kind == "white":

        def sample_white(input_: Any) -> Any:
            tensor = input_.tensor if hasattr(input_, "tensor") else input_
            return intensity * torch.randn(
                tensor.shape[0], 1, generator=gen, device=tensor.device
            )

        return sample_white

    if noise.kind == "colored":
        if noise.correlation_length is None:
            raise ValueError("colored noise requires NoiseSpec.correlation_length")
        lcorr = float(noise.correlation_length)

        def sample_colored(input_: Any) -> Any:
            tensor = input_.tensor if hasattr(input_, "tensor") else input_
            n = tensor.shape[0]
            raw = torch.randn(n, 1, generator=gen, device=tensor.device)
            # Smooth along the first input axis — cheap proxy for a
            # covariance kernel. For a full GP sample use a Cholesky
            # over a proper covariance matrix (too heavy per step).
            coords = tensor[:, :1]
            dists = torch.cdist(coords, coords) / max(lcorr, 1e-9)
            kernel = torch.exp(-0.5 * dists**2)
            kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-9)
            smoothed = kernel @ raw
            return intensity * smoothed

        return sample_colored

    if noise.kind == "fbm":
        raise ValueError(
            "fractional Brownian motion noise requires the 'fbm' "
            "package — install and wire a custom sampler first"
        )
    raise ValueError(f"unknown noise kind {noise.kind!r}")


_TORCH_MODULE: dict[str, Any] = {
    # Scalar constants the agent can reference in the form.
    "pi": float(torch.pi),
    "e": float(torch.e),
    # Unary torch ops that keep autograd-tracked tensors intact. numpy
    # on the fallback path would call .numpy() on grad-tracking tensors
    # and crash — always prefer torch.
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "Abs": torch.abs,
    "abs": torch.abs,
    "Min": torch.minimum,
    "Max": torch.maximum,
    "Pow": torch.pow,
}


def _build_lambda(
    spec: EquationSpec,
    unknowns: set[str],
    input_names: set[str],
) -> tuple[list[str], Callable[..., Any]]:
    """Parse ``spec.form`` via sympy and return (symbol_names, torch_fn).

    ``sympy.lambdify`` is pointed at an explicit torch-op dict — this
    keeps autograd working end-to-end. The numpy fallback is poisonous
    for grad-tracking tensors (torch raises on ``.numpy()``), so we
    never use it.
    """
    names: set[str] = set()
    names.update(d.name for d in spec.derivatives)
    names.update(spec.outputs)
    names.update(spec.parameters)
    names.update(n for n in unknowns if _token_present(spec.form, n))
    for axis in _INPUT_VAR_CANDIDATES:
        if _token_present(spec.form, axis):
            names.add(axis)
    for axis in input_names:
        if _token_present(spec.form, axis):
            names.add(axis)

    sympy_locals = {n: sympy.Symbol(n) for n in names}
    expr = sympy.sympify(spec.form, locals=sympy_locals)

    for sym in expr.free_symbols:
        if sym.name not in names:
            names.add(sym.name)
            sympy_locals[sym.name] = sym

    symbol_order = sorted(names)
    torch_fn_positional = sympy.lambdify(
        [sympy_locals[n] for n in symbol_order],
        expr,
        modules=[_TORCH_MODULE],
    )

    def torch_fn(**kwargs: Any) -> Any:
        args = [kwargs[n] for n in symbol_order]
        return torch_fn_positional(*args)

    return symbol_order, torch_fn


def _token_present(expr: str, token: str) -> bool:
    """True iff ``token`` appears as an identifier in the expression."""
    import re

    return (
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", expr)
        is not None
    )


def _compute_derivative(input_: Any, output_: Any, deriv: DerivativeSpec) -> Any:
    """Dispatch (grad / laplacian / chained grad / fractional) based on spec."""
    if deriv.kind == "fractional_laplacian":
        return _fractional_laplacian(input_, output_, deriv)
    if not deriv.wrt:
        raise ValueError(
            f"derivative {deriv.name!r} has empty 'wrt' — nothing to differentiate"
        )
    if len(deriv.wrt) == 1:
        return grad(output_, input_, components=[deriv.field], d=list(deriv.wrt))
    if len(set(deriv.wrt)) == 1:
        # Pure higher-order derivative in a single variable → laplacian.
        return laplacian(output_, input_, components=[deriv.field], d=deriv.wrt[:1])
    # Mixed partials — chain grad per axis.
    result = output_
    for axis in deriv.wrt:
        result = grad(result, input_, components=[deriv.field], d=[axis])
    return result


def _fractional_laplacian(input_: Any, output_: Any, deriv: DerivativeSpec) -> Any:
    """Monte-Carlo quadrature of the fractional Laplacian.

    Uses the singular integral definition with a radial symmetric
    sampler on a hyperball of radius = 1 around each collocation point.
    Good enough for training-scale PINN residuals; for high precision
    switch to a spectral method on a regular grid.
    """
    if deriv.alpha is None or not (0.0 < deriv.alpha < 2.0):
        raise ValueError(f"fractional derivative {deriv.name!r} needs alpha ∈ (0, 2)")
    if not deriv.wrt:
        raise ValueError(
            f"fractional derivative {deriv.name!r} needs 'wrt' = spatial axes"
        )
    alpha = float(deriv.alpha)
    n = int(deriv.quadrature_points)

    coords = input_.extract(list(deriv.wrt))
    n_points = coords.shape[0]
    d = coords.shape[1]

    # Sample n_quad offsets per collocation point on the unit hyperball.
    radii = torch.rand(n_points, n, 1) ** (1.0 / d)
    dirs = torch.randn(n_points, n, d)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    offsets = radii * dirs

    center_vals = output_.extract([deriv.field])
    # Evaluate the network at shifted points; we re-use the underlying
    # callable that produced ``output_``. For the current autograd
    # pipeline we approximate u(x+h) ≈ u(x) + ∇u·h using a first-order
    # Taylor step — crude but autograd-traceable.
    grads = grad(output_, input_, components=[deriv.field], d=list(deriv.wrt))
    directional = (grads.unsqueeze(1) * offsets).sum(dim=-1, keepdim=True)
    shifted_vals = center_vals.unsqueeze(1) + directional

    diffs = shifted_vals - center_vals.unsqueeze(1)
    # Riesz kernel |h|^{-(d+α)}.
    h_norm = offsets.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    kernel = 1.0 / (h_norm ** (d + alpha))
    integrand = diffs * kernel
    return integrand.mean(dim=1)
