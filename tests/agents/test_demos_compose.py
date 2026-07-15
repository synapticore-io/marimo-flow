"""Smoke-tests that the Phase B-F demo notebooks still compose.

Re-imports the `ProblemSpec`/`ControlPlan` values used by the demo
notebooks and runs the composer + one forward pass. Guards against
schema drift breaking the user-facing notebooks without the unit
tests noticing.
"""

from __future__ import annotations

import numpy as np

from marimo_flow.agents.schemas import (
    ConditionSpec,
    ControlPlan,
    ControlVariableSpec,
    DerivativeSpec,
    EquationSpec,
    ProblemSpec,
    StateSpec,
    SubdomainSpec,
)
from marimo_flow.agents.services.composer import compose_problem
from marimo_flow.control import make_rollout_surrogate, run_mpc_step


def test_ns3d_cavity_demo_composes():
    box = {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
    spec = ProblemSpec(
        name="ns3d_cavity_demo",
        output_variables=["ux", "uy", "uz", "p"],
        domain_bounds=box,
        subdomains=[
            SubdomainSpec(name="D", bounds=box),
            SubdomainSpec(name="lid", bounds={**box, "z": 1.0}),
            SubdomainSpec(name="floor", bounds={**box, "z": 0.0}),
        ],
        equations=[
            EquationSpec(
                name="mom_x",
                form="ux*ux_x + uy*ux_y + uz*ux_z + p_x - nu*(ux_xx + ux_yy + ux_zz)",
                outputs=["ux", "uy", "uz", "p"],
                derivatives=[
                    DerivativeSpec(name="ux_x", field="ux", wrt=["x"]),
                    DerivativeSpec(name="ux_y", field="ux", wrt=["y"]),
                    DerivativeSpec(name="ux_z", field="ux", wrt=["z"]),
                    DerivativeSpec(name="ux_xx", field="ux", wrt=["x", "x"]),
                    DerivativeSpec(name="ux_yy", field="ux", wrt=["y", "y"]),
                    DerivativeSpec(name="ux_zz", field="ux", wrt=["z", "z"]),
                    DerivativeSpec(name="p_x", field="p", wrt=["x"]),
                ],
                parameters={"nu": 0.1},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="mom_x"),
            ConditionSpec(subdomain="floor", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="lid", kind="fixed_value", value=1.0),
        ],
    )
    problem = compose_problem(spec)()
    assert problem.input_variables == ["x", "y", "z"]
    assert problem.output_variables == ["ux", "uy", "uz", "p"]
    assert {"D", "lid", "floor"}.issubset(problem.domains.keys())


def test_heat_rod_demo_composes_and_mpc_step_runs():
    from marimo_flow.control.heat_rod import build_heat_rod_problem_spec

    problem = compose_problem(build_heat_rod_problem_spec(alpha=0.08))()
    assert set(problem.input_variables or []) == {"x", "t", "u"}

    def predict(t_centre: float, u: float) -> float:
        return 0.85 * t_centre + 0.15 * u

    surrogate = make_rollout_surrogate(predict)

    plan = ControlPlan(
        name="heat_rod_mpc_demo",
        surrogate_uri="mem://demo",
        horizon=3,
        dt=0.05,
        controls=[ControlVariableSpec(name="u", low=-1.0, high=1.0)],
        states=[StateSpec(name="T_centre", target=0.5, weight=1.0)],
    )
    u_seq, info = run_mpc_step(plan, state_now=np.array([0.0]), surrogate=surrogate)
    assert u_seq.shape == (3, 1)
    assert np.all(np.abs(u_seq) <= 1.0 + 1e-9)
    assert info["success"]
