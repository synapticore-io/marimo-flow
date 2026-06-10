# Roadmap: beyond the current PINA presets

Last updated: 2026-06-10.

**Status**: **Phases A-0 through F shipped.** Composition-first
architecture (`services/composer.py::compose_problem`) reaches
Navier-Stokes / Maxwell / Elasticity / … the moment an agent writes a
suitable `ProblemSpec`. Inverse problems, mesh geometry, 3D plotly
viz, design optimisation, stochastic + non-local PDEs, and rolling-
horizon MPC are all in the repo with tests.

**Test baseline (2026-06-10):** 225 passed, 0 xfailed.

## Architecture pivot (2026-04-24)

The initial Phase A-0 tried to layer a persistent preset catalog on top
of the existing hardcoded `ProblemManager.create_*` factories. That
missed the point: agents could tweak parameters but could not invent
new PDE families. The user flagged it; the pivot is:

* **`services/composer.py::compose_problem`** accepts a full typed
  `ProblemSpec` (`EquationSpec` + `SubdomainSpec` + `ConditionSpec`)
  and compiles it via `sympy.lambdify` + `pina.operator.grad/laplacian`
  into a `pina.Problem` subclass at runtime.
* **No `ProblemKind` literal.** Any PDE that sympy can express and
  PINA's operators can differentiate is reachable.
* **The catalog stores compositions**, not parameter bundles. Agents
  register successful `ProblemSpec` values and clone them with
  overrides; no builtin seeding.
* **Model and solver layers keep their `Literal` kinds** (`ModelKind`,
  `SolverKind`) because those *are* finite choice spaces over PINA's
  built-in neural architectures and solver algorithms. An agent picking
  between `feedforward` / `fno` / `deeponet` is legitimate; an agent
  picking between `burgers` and `heat` is not (it should be building a
  `ProblemSpec` that encodes the physics).

Baseline after the pivot: **180 tests pass**, Burgers 1D composes and
trains end-to-end via the composer (no hardcoded preset).

Background context: [`CLAUDE.md`](../CLAUDE.md), [`README.md`](../README.md).

## Leitplanken

- **3D is the default**. Engineering problems are rarely 2D. The
  composer handles arbitrary axis sets in `domain_bounds` —
  `{"x": [...], "y": [...], "z": [...]}` makes a 3D problem, add
  `"t": [...]` for 3D+time. 2D stays as a reduction.
- **Observations come from the Data agent, never from the user.**
  `TaskSpec.observables` (what to measure) is triage-extracted.
  `ObservationSpec` (concrete `(x,y,z,t,value)` tuples) is loaded from
  file or synthesised by the Data agent — the user never hand-codes
  numpy arrays.
- **No new dependencies without a recon step**. Each phase starts with
  a recon task against the current PINA release.
- **All new deps via `uv add`**, never ad-hoc `pip install`.

## Overview

| Phase | Theme | Effort | External deps | Depends on |
|---|---|---|---|---|
| **A-0** ✅ | Preset catalog + composer + curator_toolset | — | `sympy` | — |
| **B** ✅ | Inverse path + multiphysics + Data agent | M | — | A-0 |
| **C** ✅ | Mesh/CAD geometry + 3D plotly viz | L | `meshio` (only) | — |
| **C2** | AMR (separate milestone) | XL | PINA upstream or custom | C |
| **D** ✅ | PDE-constrained optimisation | M | — (`optuna` already in) | A-0, B |
| **E** ✅ | Stochastic + non-local PDEs | M | — | — |
| **F** ✅ | Real-time control / MPC (scipy SLSQP) | L | — | D |

Effort legend: S = 1–2 sessions, M = 2–3, L = 3–4, XL = 5+.

---

## Phase A-0 — DONE

Composer-first catalog. What's shipped:

- `schemas/equation.py` — `EquationSpec`, `DerivativeSpec`, `SubdomainSpec`, `ConditionSpec`.
- `schemas/problem.py` — composition-only `ProblemSpec` (no kind enum).
- `services/composer.py` — `compose_problem(spec)` + `build_equation(spec)`.
- `services/preset_catalog.py` — DuckDB-backed store for user-authored compositions + optional YAML mirror.
- `toolsets/problem.py` — `compose_problem`, `inspect_problem`, `list_input_vars_hint`.
- `toolsets/curator.py` — `search_presets`, `describe_preset`, `register_preset`, `clone_preset`, `deprecate_preset`.
- Smoke test: 1D viscous Burgers composes + trains 2 epochs with a 8×8 FeedForward on CPU.

---

## Phase B — Inverse problems, multiphysics, Data agent — DONE (2026-04-24)

| # | Title | Shipped |
|---|---|---|
| 12 | PINA inverse-problem pattern recon | `InverseProblem` mixin + 3-arg residual auto-detect in `services/composer.py` |
| 13 | `ObservationSpec` + `UnknownParameterSpec` | `agents/schemas/observation.py` |
| 30 | Data agent: auto-ingest observations | `toolsets/data.py::load_observations_from_file` (CSV/NPZ/Parquet), `generate_synthetic_observations` |
| 14 | Composer support for inverse problems | `services/composer.py::build_equation(unknown_names=…)` + `_compile_observation` data-fitting Conditions |
| 15 | Multiphysics | two+ EquationSpec on one ProblemSpec → independent loss terms |
| 16 | Skills | `.claude/Skills/pina-inverse/`, `pina-multiphysics/`, `pina-3d/` |

Tests: `tests/agents/test_composer.py` (inverse + multiphysics paths),
`tests/agents/test_data_toolset.py` (CSV/NPZ/synthetic fixtures).

**Technical notes**

- `ObservationSpec.source ∈ {data_file, synthetic, live_sensor}`. Data
  agent dispatches:
  - `data_file` → `load_observations_from_file(path, field_name, format)`
    parses CSV / Parquet / NPZ and recognises `(x,y,z,t,field_value)`.
  - `synthetic` → `generate_synthetic_observations(n_points, true_parameters, noise_sigma)`
    runs a forward solve with the true parameters, samples n points,
    adds Gaussian noise.
  - `live_sensor` → subscribes to MQTT / Kafka (Phase F).
- `UnknownParameterSpec(name, initial, bounds, trainable)` becomes a
  `pina.LearnableParameter` inside the compiled torch callable.
- Multiphysics works without new schema: two EquationSpecs that share
  `output_variables` and both point at the interior subdomain produce
  two independent loss terms. Reference example: thermo-elasticity
  (heat + elasticity + thermal-strain coupling term).
- DuckDB extensions: tables `observations`, `unknown_parameters`.

---

## Phase C — 3D geometry, mesh, visualisation — DONE (2026-04-24)

| # | Title | Shipped |
|---|---|---|
| 17 | Non-cartesian domain recon | `pina.domain.DomainInterface` confirmed, subclassed in `MeshDomain` |
| 18 | `meshio==5.3.5` + `MeshSpec` | `uv add meshio`, `agents/schemas/mesh.py` |
| 19 | `MeshDomain` adapter | `services/mesh_domain.py` — barycentric sampling (tri/tetra/quad/hex), `is_inside`, cell-tag lookup |
| 20 | `pina-geometry` skill | `.claude/Skills/pina-geometry/` + `SubdomainSpec.mesh_ref` escape hatch |
| 31 | 3D visualisation | `core/viz3d.py` — `domain_figure` / `scatter_samples` / `volume_figure` using **plotly** (`Mesh3d`, `Volume`, `Scatter3d`, `Isosurface`). **No pyvista** — plotly was already a transitive dep via marimo, no 150 MB VTK stack. |
| 32 | Demo notebook | `examples/03_navier_stokes_3d_cavity.py` — sliders for ν / lid speed / grid / width / epochs |

Tests: `tests/agents/test_mesh_domain.py`, `tests/agents/test_viz3d.py`,
`tests/agents/test_demos_compose.py::test_ns3d_cavity_demo_composes`.

**Separate milestone**

| # | Title | Effort |
|---|---|---|
| 21 | **AMR** (adaptive mesh refinement) | XL |

**Technical notes**

- `MeshSpec(path, format, units, tags: dict[str, list[int]])` — tags
  map cell groups to physical IDs (for BCs per face).
- `MeshDomain` implements PINA's Domain interface: point-sampling via
  barycentric coords per cell, BC-subdomain lookup via mesh tags.
- `SubdomainSpec` gains an optional `mesh_ref` escape hatch so
  conditions can target mesh tags instead of cartesian bounds.
- CAD: `pyvista` (STL/VTK/GLTF native) + optional `pythonOCC` for
  STEP/IGES. Both soft-deps via `uv add --optional`.
- AMR: `RBAPINN` already covers adaptive sampling. Real h/p-AMR needs
  either PINA upstream support or a custom refinement loop.

---

## Phase D — PDE-constrained optimisation — DONE (2026-04-24)

| # | Title | Shipped |
|---|---|---|
| 22 | Design agent + schemas | `agents/schemas/optimization.py` — `DesignVariableSpec`, `ConstraintSpec`, `OptimizationPlan`; `"design"` added to `AgentRole` |
| 23 | `design_toolset` | `toolsets/design.py` — `build_optimization_plan`, `apply_overrides`, `evaluate_constraints`, `run_design_sweep` (Optuna TPE / scipy SLSQP); penalty + augmented-Lagrangian via `services/design.py::ConstraintAggregator` |

Tests: `tests/agents/test_design.py` (apply_overrides + constraint
aggregator + toolset roundtrip).

**Technical notes**

- New agent role `design`.
- `OptimizationPlan(objective, design_variables, constraints, method)`
  with `method ∈ {optuna_tpe, scipy_slsqp, penalty, augmented_lagrangian}`.
- Constraint handler: penalty (`loss_total = loss_pde + λ·max(0, g(x))²`)
  vs. augmented-Lagrangian (λ updates per outer loop).
- Smoke-test: 2D/3D topology optimisation of a plate under load
  (minimal material at a stiffness constraint).

---

## Phase E — Stochastic + non-local PDEs — DONE (2026-04-24)

| # | Title | Shipped |
|---|---|---|
| 24 | Stochastic PDEs | `agents/schemas/stochastic.py::NoiseSpec` (white / colored / fbm); composer wraps residual with additive torch noise via `_build_noise_sampler` |
| 25 | Non-local / fractional | `DerivativeKind` literal on `DerivativeSpec` + `alpha`, `quadrature_points`; `services/composer.py::_fractional_laplacian` uses autograd-traceable Riesz-kernel Monte-Carlo quadrature |

Tests: `tests/agents/test_composer.py` (noise + fractional paths).

**Technical notes**

- `NoiseSpec(type="white|colored|fbm", intensity, correlation)` extends
  the composer to wrap an EquationSpec with an additive noise term.
  Solver: Monte-Carlo over noise realisations with a mean PINN loss, or
  variational PINN.
- Fractional Laplace as a custom `pina.Equation` with spectral or
  diffusive-representation quadrature. Expose via a new derivative kind
  in the composer. Test against an analytical fractional-Poisson
  solution on an interval.

---

## Phase F — Real-time control / MPC — DONE (2026-04-24)

| # | Title | Shipped |
|---|---|---|
| 26 | MPC library recon | **scipy SLSQP** chosen over do-mpc/cvxpy/acados (no casadi / C++ stack). Rationale: for scalar-state lab demos the existing `scipy.optimize.minimize` is enough; do-mpc pulls ~50 MB casadi. |
| 27 | `control/` module + schemas | `src/marimo_flow/control/` — `run_mpc_step`, `simulate_closed_loop` (rolling-horizon, warm-started); `agents/schemas/control.py::ControlPlan`, `ControlVariableSpec`, `StateSpec`; `"control"` added to `AgentRole`; `toolsets/control.py` exposes `build_control_plan`, `mpc_step`, `closed_loop_simulation` |
| 28 | Closed-loop demo | `examples/04_mpc_heat_rod.py` — trains 1D heat surrogate, steers centre-temperature toward a setpoint |

Tests: `tests/agents/test_control.py` (single-step + rollout),
`tests/agents/test_demos_compose.py::test_heat_rod_demo_composes_and_mpc_step_runs`.

**Technical notes**

- `src/marimo_flow/control/` parallel to `core/` and `agents/` — keeps
  the control domain from leaking into the PINN kernel.
- `ControlPlan(horizon, objective, constraints, surrogate_uri)`. The
  `surrogate_uri` points at a trained solver in the DuckDB `artifacts`
  table.
- Default library: `do-mpc` (high-level, CasADi-based). `cvxpy` only
  where the problem is convex. `acados` (C++) is out of scope for this
  repo — sibling project material.
- Smoke-test: 1D heat-rod stabilisation or inverted pendulum with a
  PINN surrogate as the `dynamics_function`.

---

## How to reproduce / verify

```bash
# Full test suite — must stay at 225 passed, 0 xfailed
uv run pytest -q

# Smoke-run the new demos outside marimo
uv run pytest tests/agents/test_demos_compose.py -v

# Launch demos interactively
uv run marimo edit examples/03_navier_stokes_3d_cavity.py
uv run marimo edit examples/04_mpc_heat_rod.py

# Inspect provenance (all 13 DuckDB tables)
uv run marimo edit examples/02_provenance_dashboard.py
```

## Non-goals (still)

- **Shock-fitting / Riemann solvers** for hyperbolic conservation laws.
  PINN is structurally poor at this; use FV/DG (dolfinx, Clawpack,
  OpenFOAM) in a separate project.
- **Volumetric rendering of tera-scale fields**. plotly covers lab
  scale; Paraview / Catalyst stays outside.
- **Real HPC parallelism**. Lightning covers DDP; SLURM / Horovod go
  out of scope.
- **AMR (Phase C2)** — `RBAPINN` covers adaptive sampling today; real
  h/p-AMR still blocks on PINA upstream support.

## Open milestones

| # | Milestone | Status |
|---|---|---|
| 21 | Adaptive mesh refinement (Phase C2, XL) | not started |
| — | MPC beyond scalar state (requires `do-mpc` / `acados`) | escalate before `uv add` |
| — | CAD bridge (STEP / IGES via OCCT) | not yet needed |

---

All 20 original roadmap tasks (B-F) completed in Phase B-F commits.
Next natural step: decide whether C2 (AMR) is worth pulling in PINA
upstream changes, or whether the `RBAPINN` adaptive-sampling path
already covers the demands.
