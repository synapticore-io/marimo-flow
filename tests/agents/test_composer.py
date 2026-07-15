"""Tests for the PDE composer (compose_problem + build_equation).

Uses real sympy + PINA operators — no mocks. Verifies that composed
problems have the right structure (output_vars, subdomains, time-dependency)
and that the symbolic residual produces a torch tensor with the expected
autograd signal.
"""

from __future__ import annotations

import pytest
import torch
from pina.problem import SpatialProblem, TimeDependentProblem

from marimo_flow.agents.schemas import (
    ConditionSpec,
    DerivativeSpec,
    EquationSpec,
    ObservationSpec,
    ProblemSpec,
    SubdomainSpec,
    UnknownParameterSpec,
)
from marimo_flow.agents.services.composer import build_equation, compose_problem


def _burgers_spec(name: str = "burgers_1d") -> ProblemSpec:
    return ProblemSpec(
        name=name,
        output_variables=["u"],
        domain_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="left", bounds={"x": -1.0, "t": [0.0, 1.0]}),
            SubdomainSpec(name="right", bounds={"x": 1.0, "t": [0.0, 1.0]}),
            SubdomainSpec(name="t0", bounds={"x": [-1.0, 1.0], "t": 0.0}),
            SubdomainSpec(name="D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(
                name="burgers",
                form="u_t + u*u_x - nu*u_xx",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_t", field="u", wrt=["t"]),
                    DerivativeSpec(name="u_x", field="u", wrt=["x"]),
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                ],
                parameters={"nu": 0.01},
            ),
            EquationSpec(
                name="ic",
                form="u + sin(pi*x)",
                outputs=["u"],
                derivatives=[],
                parameters={"pi": 3.141592653589793},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="left", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="right", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="t0", kind="equation", equation_name="ic"),
            ConditionSpec(subdomain="D", kind="equation", equation_name="burgers"),
        ],
    )


def test_compose_burgers_is_time_dependent():
    cls = compose_problem(_burgers_spec())
    assert issubclass(cls, TimeDependentProblem)
    assert issubclass(cls, SpatialProblem)
    assert cls.__name__ == "burgers_1d"


def test_compose_stationary_poisson_2d():
    spec = ProblemSpec(
        name="poisson_2d",
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="D", bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(
                name="poisson",
                form="u_xx + u_yy + sin(pi*x)*sin(pi*y)",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                    DerivativeSpec(name="u_yy", field="u", wrt=["y", "y"]),
                ],
                parameters={"pi": 3.141592653589793},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="poisson"),
        ],
    )
    cls = compose_problem(spec)
    assert issubclass(cls, SpatialProblem)
    assert not issubclass(cls, TimeDependentProblem)
    assert "D" in cls.domains


def test_compose_poisson_3d_unit_cube():
    """3D problems work without any code changes — just more axes."""
    spec = ProblemSpec(
        name="poisson_3d",
        output_variables=["u"],
        domain_bounds={
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
        },
        subdomains=[
            SubdomainSpec(
                name="D",
                bounds={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
            ),
        ],
        equations=[
            EquationSpec(
                name="poisson3d",
                form="u_xx + u_yy + u_zz",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                    DerivativeSpec(name="u_yy", field="u", wrt=["y", "y"]),
                    DerivativeSpec(name="u_zz", field="u", wrt=["z", "z"]),
                ],
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="poisson3d"),
        ],
    )
    cls = compose_problem(spec)
    assert cls.output_variables == ["u"]
    # PINA exposes input_variables on instances (derived from the
    # CartesianDomain axes), not on the class — instantiate to check.
    instance = cls()
    assert set(instance.input_variables or []) == {"x", "y", "z"}


def test_compose_rejects_unknown_equation_reference():
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        equations=[],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="nope"),
        ],
    )
    with pytest.raises(ValueError, match="nope"):
        compose_problem(spec)


def test_compose_rejects_fixed_value_without_value():
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        conditions=[ConditionSpec(subdomain="D", kind="fixed_value")],
    )
    with pytest.raises(ValueError, match="fixed_value"):
        compose_problem(spec)


def test_build_equation_returns_pina_equation():
    """Happy-path build — the returned object is a pina.Equation wrapper."""
    from pina.equation import Equation

    spec = EquationSpec(
        name="linear",
        form="u - (2.0*x + 1.0)",
        outputs=["u"],
        derivatives=[],
        parameters={},
    )
    eq = build_equation(spec)
    assert isinstance(eq, Equation)


def test_build_equation_inline_in_condition():
    """Inline EquationSpec on a ConditionSpec resolves at compose time."""
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        conditions=[
            ConditionSpec(
                subdomain="D",
                kind="equation",
                equation_inline=EquationSpec(
                    name="inline",
                    form="u - 3.0",
                    outputs=["u"],
                    derivatives=[],
                    parameters={},
                ),
            ),
        ],
    )
    cls = compose_problem(spec)
    assert "D" in cls.conditions


def test_composed_burgers_trains_end_to_end():
    """Full integration: compose → pick model → pick solver → train 2 epochs.

    Verifies the composer's torch callable is autograd-compatible and that
    PINA's Trainer can actually optimise against it without any hardcoded
    PDE factory on the path.
    """
    from marimo_flow.core import ModelManager, SolverManager, train_solver

    cls = compose_problem(_burgers_spec("burgers_train_smoke"))
    problem = cls()
    model = ModelManager.create("feedforward", problem=problem, layers=[8, 8])
    solver = SolverManager.create(
        "pinn", problem=problem, model=model, learning_rate=1e-3
    )
    trainer = train_solver(
        solver, max_epochs=2, accelerator="cpu", n_points=64, sample_mode="random"
    )
    metrics = trainer.callback_metrics
    # Each condition gets its own loss term — they all evaluated.
    for key in ("left_loss", "right_loss", "t0_loss", "D_loss"):
        assert key in metrics, f"missing {key} in {dict(metrics)}"
        assert torch.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"


def test_inverse_problem_mixes_in_inverse_base_and_sets_parameter_domain():
    """A ProblemSpec with unknowns produces a SpatialProblem + InverseProblem."""
    from pina.problem import InverseProblem

    spec = ProblemSpec(
        name="inverse_burgers",
        output_variables=["u"],
        domain_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(
                name="burgers",
                form="u_t + u*u_x - nu*u_xx",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_t", field="u", wrt=["t"]),
                    DerivativeSpec(name="u_x", field="u", wrt=["x"]),
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                ],
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="burgers"),
        ],
        unknowns=[UnknownParameterSpec(name="nu", low=0.001, high=0.1)],
    )
    cls = compose_problem(spec)
    assert issubclass(cls, InverseProblem)
    assert "nu" in cls.unknown_parameter_domain.variables


def test_inverse_residual_pulls_unknown_from_params_not_parameters():
    """When a symbol is declared unknown, build_equation emits a 3-arg residual."""
    spec = EquationSpec(
        name="poisson_inverse",
        form="u_xx + mu",
        outputs=["u"],
        derivatives=[DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"])],
    )
    eq = build_equation(spec, unknown_names={"mu"})
    import inspect

    assert len(inspect.signature(eq._Equation__equation).parameters) == 3


def test_direct_residual_stays_two_arg_when_no_unknowns():
    spec = EquationSpec(
        name="poisson",
        form="u_xx + sin(pi*x)",
        outputs=["u"],
        derivatives=[DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"])],
        parameters={"pi": 3.141592653589793},
    )
    eq = build_equation(spec)
    import inspect

    assert len(inspect.signature(eq._Equation__equation).parameters) == 2


def test_multiphysics_two_equations_same_subdomain():
    """Two EquationSpecs pointing at the same interior subdomain coexist.

    Verifies the composer emits independent Conditions for each — the
    loss has one term per equation key.
    """
    spec = ProblemSpec(
        name="thermoelastic_1d",
        output_variables=["T", "uvel"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        equations=[
            EquationSpec(
                name="heat",
                form="T_xx",
                outputs=["T"],
                derivatives=[DerivativeSpec(name="T_xx", field="T", wrt=["x", "x"])],
            ),
            EquationSpec(
                name="elastic",
                form="uvel_xx - alpha*T",
                outputs=["uvel", "T"],
                derivatives=[
                    DerivativeSpec(name="uvel_xx", field="uvel", wrt=["x", "x"])
                ],
                parameters={"alpha": 0.5},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="heat"),
        ],
    )
    # A single Condition per subdomain is the PINA semantics. To couple
    # heat+elastic on the same subdomain we must either have two
    # subdomains pointing at the same CartesianDomain (canonical)
    # or rely on a SystemEquation. Here we assert the first pattern
    # composes correctly — the canonical way to express multiphysics.
    spec.subdomains.append(SubdomainSpec(name="D2", bounds={"x": [0.0, 1.0]}))
    spec.conditions.append(
        ConditionSpec(subdomain="D2", kind="equation", equation_name="elastic")
    )
    cls = compose_problem(spec)
    assert "D" in cls.conditions
    assert "D2" in cls.conditions


def test_observation_condition_materialised_points():
    """ObservationSpec with filled points/values compiles to a data Condition."""
    spec = ProblemSpec(
        name="poisson_with_data",
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        equations=[
            EquationSpec(
                name="poisson",
                form="u_xx",
                outputs=["u"],
                derivatives=[DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"])],
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="poisson"),
        ],
        observations=[
            ObservationSpec(
                name="sensor_grid",
                field="u",
                axes=["x"],
                points=[[0.1], [0.5], [0.9]],
                values=[[0.05], [0.25], [0.45]],
            ),
        ],
    )
    cls = compose_problem(spec)
    assert "sensor_grid" in cls.conditions


def test_observation_without_materialisation_errors():
    """An un-materialised observation is rejected before we touch PINA."""
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0]},
        subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
        conditions=[
            ConditionSpec(subdomain="D", kind="fixed_value", value=0.0),
        ],
        observations=[
            ObservationSpec(name="sensor", field="u", axes=["x"]),
        ],
    )
    with pytest.raises(ValueError, match="not materialised"):
        compose_problem(spec)


def test_compose_heat_rod_parametric_control():
    from marimo_flow.control.heat_rod import build_heat_rod_problem_spec

    cls = compose_problem(build_heat_rod_problem_spec(alpha=0.08))
    instance = cls()
    assert set(instance.input_variables or []) == {"x", "t", "u"}
    assert instance.output_variables == ["T"]
    assert "right" in instance.domains


def test_compose_emits_meaningful_class_name():
    cls = compose_problem(_burgers_spec(name="my_special_burgers"))
    assert cls.__name__ == "my_special_burgers"

    anon = compose_problem(
        ProblemSpec(
            output_variables=["u"],
            domain_bounds={"x": [0.0, 1.0]},
            subdomains=[SubdomainSpec(name="D", bounds={"x": [0.0, 1.0]})],
            conditions=[ConditionSpec(subdomain="D", kind="fixed_value", value=0.0)],
        )
    )
    assert anon.__name__ == "ComposedProblem"
