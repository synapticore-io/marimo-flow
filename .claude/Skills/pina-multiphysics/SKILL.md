---
name: pina-multiphysics
description: Compose coupled multi-field PINA problems (e.g. thermo-elasticity, magnetohydrodynamics) by listing multiple EquationSpecs on one ProblemSpec — no new schema needed.
triggers:
  - multiphysics
  - coupled pde
  - thermo-elastic
  - magnetohydrodynamics
  - mhd
  - reaction-diffusion
  - navier-stokes thermal
  - phase field
---

# PINA Multiphysics Composition

Use this skill when the user's problem has **more than one PDE coupled
through shared fields**. Typical examples:

- **Thermo-elasticity** — heat equation + elasticity + thermal-strain
  source term.
- **Reaction-diffusion** — two species coupled via non-linear reaction.
- **Navier-Stokes with heat transfer** — momentum + continuity + heat
  equation with buoyancy.
- **MHD** — Navier-Stokes + Maxwell with Lorentz-force coupling.

## Key insight: no new schema

The existing ``ProblemSpec`` already supports it. Just list **multiple
`EquationSpec` entries** that share ``output_variables`` and point each
one at a distinct interior subdomain. PINA emits one loss term per
condition; the shared output network couples them through gradient
descent.

## Recipe

1. In ``output_variables`` list **every** field that any equation
   references: e.g. ``["T", "ux", "uy"]`` for 2D thermo-elasticity.
2. Create **one interior subdomain per equation** — PINA keys conditions
   on subdomain names, so you cannot have two conditions on the same
   name. Keep the actual bounds identical; only the names differ:

   ```python
   SubdomainSpec(name="D_heat",    bounds={"x": [0,1], "y": [0,1]}),
   SubdomainSpec(name="D_elastic", bounds={"x": [0,1], "y": [0,1]}),
   ```

   Alternative with one shared point set: a single interior subdomain
   with ``ConditionSpec(subdomain="D", kind="equation",
   equation_names=["heat", "elastic"])`` (PINA ``SystemEquation``). The
   3D Navier–Stokes demo (``examples/03_navier_stokes_3d_cavity.py``)
   uses this for momentum x/y/z + continuity. For multi-output walls use
   ``kind="fixed_value", values={"ux": 0.0, "uy": 0.0}`` — a bare
   ``value`` would pin every output, pressure included.

3. Declare each PDE as its own ``EquationSpec``. List in ``outputs``
   only the fields that appear in the **coupling term** of that
   equation — the composer only extracts those from ``output_``.

4. Attach each equation to its subdomain with one
   ``ConditionSpec(kind="equation", equation_name=...)``.

5. Boundary / initial conditions stay as before; reuse the same wall
   subdomains for every field (``FixedValue`` applies to each declared
   output component automatically).

## Worked example — 2D thermo-elasticity

```python
from marimo_flow.agents.schemas import (
    ConditionSpec, DerivativeSpec, EquationSpec,
    ProblemSpec, SubdomainSpec,
)

spec = ProblemSpec(
    name="thermoelastic_plate",
    output_variables=["T", "ux", "uy"],
    domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    subdomains=[
        SubdomainSpec(name="bottom", bounds={"x": [0,1], "y": 0.0}),
        SubdomainSpec(name="top",    bounds={"x": [0,1], "y": 1.0}),
        SubdomainSpec(name="left",   bounds={"x": 0.0,   "y": [0,1]}),
        SubdomainSpec(name="right",  bounds={"x": 1.0,   "y": [0,1]}),
        SubdomainSpec(name="D_heat",    bounds={"x": [0,1], "y": [0,1]}),
        SubdomainSpec(name="D_elastic", bounds={"x": [0,1], "y": [0,1]}),
    ],
    equations=[
        EquationSpec(
            name="heat",
            form="T_xx + T_yy",
            outputs=["T"],
            derivatives=[
                DerivativeSpec(name="T_xx", field="T", wrt=["x","x"]),
                DerivativeSpec(name="T_yy", field="T", wrt=["y","y"]),
            ],
        ),
        EquationSpec(
            name="elastic_x",
            form="ux_xx + ux_yy - alpha*T",
            outputs=["ux", "T"],                 # COUPLING — T appears here
            derivatives=[
                DerivativeSpec(name="ux_xx", field="ux", wrt=["x","x"]),
                DerivativeSpec(name="ux_yy", field="ux", wrt=["y","y"]),
            ],
            parameters={"alpha": 0.5},
        ),
    ],
    conditions=[
        ConditionSpec(subdomain="bottom", kind="fixed_value", value=0.0),
        ConditionSpec(subdomain="top",    kind="fixed_value", value=0.0),
        ConditionSpec(subdomain="left",   kind="fixed_value", value=0.0),
        ConditionSpec(subdomain="right",  kind="fixed_value", value=0.0),
        ConditionSpec(subdomain="D_heat",    kind="equation", equation_name="heat"),
        ConditionSpec(subdomain="D_elastic", kind="equation", equation_name="elastic_x"),
    ],
)
cls = compose_problem(spec)
```

## Choosing the network architecture

Pick a **single feed-forward / FNO / DeepONet with output dim =
len(output_variables)** so the fields share representation capacity.
The Model agent does this automatically when it sees multiple outputs.

## Validation expectations

- Each equation gets its own loss term in the trainer metrics
  (``D_heat_loss``, ``D_elastic_loss``, …).
- If one loss term plateaus while another keeps improving, add
  loss-weights via ``SolverPlan.weights`` (not yet wired in Phase A-0 —
  escalate if needed).

## When to split into two problems instead

Couple **via forward time stepping** if the physics are weakly coupled
and run at different timescales (e.g. fast chemistry + slow flow).
Multiphysics composition is ideal for tightly coupled steady-state or
homogeneous-timescale problems.
