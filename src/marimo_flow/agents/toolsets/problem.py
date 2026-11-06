"""FunctionToolset for the Problem agent — composition-first.

The single core tool ``compose_problem`` accepts a full ``ProblemSpec``
(equations + subdomains + conditions) and runs it through
``services.composer.compose_problem`` to build a live ``pina.Problem``
subclass. No hardcoded ``kind`` enum — any PDE that PINA's operators
can express is reachable.

Companion tools:

* ``list_input_vars_hint`` — reminds the agent which axis names
  (``x``, ``y``, ``z``, ``t``) the composer recognises.
* ``inspect_problem`` — introspects a freshly composed problem so the
  agent can sanity-check the domain + conditions before handoff.
"""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import ProblemSpec
from marimo_flow.agents.services.composer import compose_problem as _compose
from marimo_flow.agents.toolsets._registry import register_artifact, require_state

problem_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="problem")


@problem_toolset.tool
def list_input_vars_hint(ctx: RunContext[FlowDeps]) -> list[str]:  # noqa: ARG001
    """Axis names the composer understands out of the box.

    Anything else (``r``, ``theta``, ``phi`` for radial coords) can still
    be used in equations, but PINA's cartesian grad/laplacian operate
    on these standard names — if your geometry needs other axes, use a
    subdomain transform or stick to this set.
    """
    return ["x", "y", "z", "t"]


@problem_toolset.tool
def compose_problem(
    ctx: RunContext[FlowDeps],
    spec: dict[str, Any],
) -> str:
    """Build a PINA Problem from a typed ``ProblemSpec`` and register it.

    Args:
        spec: dict that validates as a ``ProblemSpec``. Key fields:

            * ``output_variables``: list[str], e.g. ``["u"]``.
            * ``domain_bounds``: dict[str, [min, max]] — the ambient
              domain. Presence of ``"t"`` promotes to TimeDependentProblem.
            * ``subdomains``: list of ``{name, bounds}`` entries.
            * ``equations``: list of ``EquationSpec`` entries, each with
              ``form`` (sympy expression), ``outputs``, ``derivatives``
              (explicit name→field+wrt), and ``parameters`` (scalar).
            * ``conditions``: list of ``{subdomain, kind, ...}`` entries
              mapping each subdomain to either a ``fixed_value`` (Dirichlet;
              ``value`` pins every output, ``values={field: v}`` only the
              named ones) or an ``equation`` (interior residual / custom
              BC/IC; ``equation_names=[...]`` enforces a coupled system).
            * ``name``: optional label, reused if the agent later
              registers this composition as a preset.

    Returns the MLflow artifact URI of the composed problem; the live
    class lives in ``deps.registry`` under that URI for the Model /
    Solver agents to consume.

    Validation errors (unknown axes, undefined equation refs, empty
    derivatives) come back as ``ModelRetry`` so the LLM can fix the
    spec and try again.
    """
    state = require_state(ctx.deps)
    try:
        validated = ProblemSpec.model_validate(spec)
    except Exception as exc:
        raise ModelRetry(
            f"ProblemSpec failed validation: {exc}. Check field names + types."
        ) from exc
    try:
        problem_cls = _compose(validated)
    except Exception as exc:
        raise ModelRetry(
            f"compose_problem failed: {exc}. Common causes: (a) equation "
            "form references undeclared symbol, (b) condition points at "
            "unknown equation_name, (c) subdomain axis not in domain_bounds."
        ) from exc

    problem = problem_cls()
    uri = register_artifact(
        deps=ctx.deps,
        state=state,
        artifact_path="problem",
        filename="problem_spec.json",
        record=validated.model_dump(mode="json"),
        instance=problem,
    )
    state.problem_artifact_uri = uri
    state.problem_spec = validated
    with contextlib.suppress(Exception):
        task_id = state.task_spec.task_id if state.task_spec else "unknown"
        ctx.deps.provenance().record_problem_spec(task_id, validated)
    return uri


@problem_toolset.tool
def inspect_problem(ctx: RunContext[FlowDeps]) -> dict[str, Any]:
    """Return a summary of the currently registered problem.

    Useful for the Problem agent (after compose) and the Model / Solver
    agents (before build) to verify the domain + conditions look right
    without having to dump the raw PINA object.
    """
    state = require_state(ctx.deps)
    if state.problem_artifact_uri is None:
        raise ModelRetry("No problem registered yet — call compose_problem first.")
    problem = ctx.deps.registry[state.problem_artifact_uri]
    summary: dict[str, Any] = {
        "class_name": type(problem).__name__,
        "output_variables": list(problem.output_variables or []),
        "input_variables": list(problem.input_variables or []),
        "subdomains": list((problem.domains or {}).keys()),
        "conditions": list((problem.conditions or {}).keys()),
    }
    if state.problem_spec is not None:
        summary["equations"] = [
            {
                "name": eq.name,
                "form": eq.form,
                "parameters": eq.parameters,
            }
            for eq in state.problem_spec.equations
        ]
    return summary
