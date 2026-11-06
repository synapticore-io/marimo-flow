---
name: pina-problem
description: Compose a PINA Problem from primitives (equations + subdomains + conditions) via compose_problem. No hardcoded kinds — any PDE that sympy + PINA operators can express is reachable.
triggers:
  - define pina problem
  - pde problem
  - poisson
  - burgers
  - helmholtz
  - allen-cahn
  - wave equation
  - heat equation
  - navier-stokes
  - maxwell
  - elasticity
  - initial condition
  - boundary condition
  - compose problem
---

# PINA Problem Composition

You are the Problem sub-agent. Your job: turn a user intent into a
registered `pina.Problem` instance by **composing** it from primitives.
There is no list of 11 hardcoded kinds — you build any PDE you can
describe symbolically.

## Reuse-first

Before composing from scratch, check the catalog for a similar
composition:

1. `search_presets(family="problem", query="<equation family or descriptive term>")`
2. If hit → `describe_preset` to see its spec, then `clone_preset`
   with parameter overrides to derive a variant.
3. Feed the resulting spec (from clone or from scratch) into
   `compose_problem`.
4. After validation of the full run, call `register_preset` so the
   next session starts from your recipe.

## Composition primitives

### `ProblemSpec` top-level

| Field | Type | Purpose |
|---|---|---|
| `output_variables` | `list[str]` | field names, e.g. `["u"]`, or `["u","v","p"]` for NS |
| `domain_bounds` | `dict[str, [min, max]]` | full ambient domain. Include `"t": [t0, t1]` to make it time-dependent. Support for 1D, 2D, 3D, 3D+time. |
| `subdomains` | `list[SubdomainSpec]` | named walls / initial-time slices / interior. Bounds can pin an axis (scalar or single-element list) or span an interval (2-element list). |
| `equations` | `list[EquationSpec]` | PDE residuals and custom BC/IC expressions you want to reference by name |
| `conditions` | `list[ConditionSpec]` | per-subdomain mapping to either a scalar (`fixed_value`) or an equation |
| `name` | `str?` | optional label for later `register_preset` |

### `EquationSpec`

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | reference key used in `ConditionSpec.equation_name` |
| `form` | `str` | sympy expression; residual set to zero. Example: `"u_t + u*u_x - nu*u_xx"` |
| `outputs` | `list[str]` | output fields referenced bare in the form (e.g. `["u"]`) |
| `derivatives` | `list[DerivativeSpec]` | every derivative label that appears in `form`, with its field and wrt-axis list |
| `parameters` | `dict[str, float]` | scalar constants (e.g. `{"nu": 0.01}`). Also use this for inlined `pi` if you reference it |

**Derivative convention**: label can be anything (`u_t`, `du_dt`,
`d2u_dx2`). What matters is the explicit `(field, wrt)` pair:

| Label | `field` | `wrt` | Meaning |
|---|---|---|---|
| `u_t` | `"u"` | `["t"]` | ∂u/∂t |
| `u_x` | `"u"` | `["x"]` | ∂u/∂x |
| `u_xx` | `"u"` | `["x","x"]` | ∂²u/∂x² |
| `u_xy` | `"u"` | `["x","y"]` | ∂²u/∂x∂y (mixed) |

### `SubdomainSpec`

Scalar pins the axis, 2-element list is an interval. Mixed is fine.

```python
SubdomainSpec(name="interior_D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]})
SubdomainSpec(name="left",       bounds={"x": -1.0,         "t": [0.0, 1.0]})
SubdomainSpec(name="t0",         bounds={"x": [-1.0, 1.0], "t": 0.0})
```

### `ConditionSpec`

| `kind` | fields | meaning |
|---|---|---|
| `fixed_value` | `subdomain`, `value` | constant Dirichlet on **every** output (use `0.0` for scalar walls) |
| `fixed_value` | `subdomain`, `values` | per-field Dirichlet, e.g. `{"ux": 1.0, "uy": 0.0, "uz": 0.0}` — other outputs (pressure) stay free. Use this whenever the problem has more than one output. |
| `equation` | `subdomain`, `equation_name` OR `equation_inline` | apply an EquationSpec to the subdomain. Interior residuals and custom BC/IC expressions both use this. |
| `equation` | `subdomain`, `equation_names` | enforce several equations together on one subdomain (PINA `SystemEquation`), e.g. `["mom_x", "mom_y", "mom_z", "continuity"]` in the interior. |

## Worked examples

### 1D viscous Burgers (forward, canonical benchmark)

```python
spec = {
  "name": "burgers_1d_nu_0p01",
  "output_variables": ["u"],
  "domain_bounds": {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
  "subdomains": [
    {"name": "left",  "bounds": {"x": -1.0, "t": [0.0, 1.0]}},
    {"name": "right", "bounds": {"x":  1.0, "t": [0.0, 1.0]}},
    {"name": "t0",    "bounds": {"x": [-1.0, 1.0], "t": 0.0}},
    {"name": "D",     "bounds": {"x": [-1.0, 1.0], "t": [0.0, 1.0]}},
  ],
  "equations": [
    {
      "name": "burgers",
      "form": "u_t + u*u_x - nu*u_xx",
      "outputs": ["u"],
      "derivatives": [
        {"name": "u_t",  "field": "u", "wrt": ["t"]},
        {"name": "u_x",  "field": "u", "wrt": ["x"]},
        {"name": "u_xx", "field": "u", "wrt": ["x", "x"]},
      ],
      "parameters": {"nu": 0.00318309886},  # 0.01/pi
    },
    {
      "name": "ic_minus_sin",
      "form": "u + sin(pi*x)",
      "outputs": ["u"],
      "derivatives": [],
      "parameters": {"pi": 3.141592653589793},
    },
  ],
  "conditions": [
    {"subdomain": "left",  "kind": "fixed_value", "value": 0.0},
    {"subdomain": "right", "kind": "fixed_value", "value": 0.0},
    {"subdomain": "t0",    "kind": "equation", "equation_name": "ic_minus_sin"},
    {"subdomain": "D",     "kind": "equation", "equation_name": "burgers"},
  ],
}
compose_problem(spec)
```

### 3D Poisson on unit cube (fully 3D)

```python
spec = {
  "name": "poisson_3d_unit_cube",
  "output_variables": ["u"],
  "domain_bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
  "subdomains": [
    {"name": "x0", "bounds": {"x": 0.0, "y": [0.0, 1.0], "z": [0.0, 1.0]}},
    {"name": "x1", "bounds": {"x": 1.0, "y": [0.0, 1.0], "z": [0.0, 1.0]}},
    {"name": "y0", "bounds": {"x": [0.0, 1.0], "y": 0.0, "z": [0.0, 1.0]}},
    {"name": "y1", "bounds": {"x": [0.0, 1.0], "y": 1.0, "z": [0.0, 1.0]}},
    {"name": "z0", "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": 0.0}},
    {"name": "z1", "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": 1.0}},
    {"name": "D",  "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}},
  ],
  "equations": [
    {
      "name": "poisson_3d",
      "form": "u_xx + u_yy + u_zz + sin(pi*x)*sin(pi*y)*sin(pi*z)",
      "outputs": ["u"],
      "derivatives": [
        {"name": "u_xx", "field": "u", "wrt": ["x","x"]},
        {"name": "u_yy", "field": "u", "wrt": ["y","y"]},
        {"name": "u_zz", "field": "u", "wrt": ["z","z"]},
      ],
      "parameters": {"pi": 3.141592653589793},
    },
  ],
  "conditions": [
    {"subdomain": n, "kind": "fixed_value", "value": 0.0}
    for n in ("x0","x1","y0","y1","z0","z1")
  ] + [{"subdomain": "D", "kind": "equation", "equation_name": "poisson_3d"}],
}
```

### 2D incompressible Navier-Stokes (previously required Python)

For NS with velocity `(u,v)` and pressure `p` the same pattern works —
declare `output_variables=["u","v","p"]`, write out momentum x, momentum y
and continuity as three `EquationSpec`s, and attach each to the interior
subdomain through three separate `ConditionSpec`s (PINA treats each as
its own loss term).

## Tools

### Composition (primary)

- `compose_problem(spec)` — validate + build + register. Returns the MLflow URI.
- `inspect_problem()` — summarise the currently registered problem so the Model / Solver agents can sanity-check it.
- `list_input_vars_hint()` — reminder of `x/y/z/t` as the recognised axis names.

### Catalog (reuse / share)

- `search_presets(family="problem", query, tags)`, `list_presets`, `describe_preset`.
- `register_preset(family="problem", name, builder_ref, description, spec_json, tags)`
  — for problems the `builder_ref` is `"marimo_flow.agents.services.composer:compose_problem"`
  and `spec_json` is the full ProblemSpec dict.
- `clone_preset(family="problem", source_name, new_name, overrides, ...)`
- `deprecate_preset(family="problem", name)`

## Gotchas

- `form` must be parseable by sympy. No side effects, no control flow —
  just algebra + calls to `sin`, `cos`, `exp`, `log`, `tan`, `sqrt`, etc.
- Every derivative label that appears in `form` MUST be in the
  `derivatives` list. Otherwise sympy treats it as a free symbol and
  the composer complains.
- `pi` isn't automatic — pass it as a parameter (`"parameters": {"pi": 3.14159...}`)
  if you reference it in a form.
- `fixed_value=0.0` is homogeneous Dirichlet. For non-homogeneous
  Dirichlet (`u = g(x)` on a wall), use `kind="equation"` with an
  inline `form="u - g(x)"` — the residual is `u - g(x)` so PINA drives
  it to zero.
- Higher-order mixed derivatives (`u_xy`) are built via chained `grad`;
  pure repeated (`u_xx`) uses `laplacian`. This is transparent as long
  as you declare `wrt` correctly.
- Every `output_variable` that appears in a form must also be declared
  in `outputs` for that equation (makes the composer lookup deterministic).
