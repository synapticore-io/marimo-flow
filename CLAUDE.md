# CLAUDE.md

## Project: marimo-flow

Reactive Python notebooks with MLflow tracking and PINA physics-informed neural networks.

## Commands

- `uv sync` - Install dependencies
- `marimo edit examples/<name>.py` - Open notebook
- `mlflow ui` - Start MLflow dashboard
- `ruff format . && ruff check --fix .` - Format/lint

## MCP Servers

- **marimo**: start manually with `marimo edit --mcp --no-token --port 2718 --headless`
- **mlflow**: `mlflow mcp run` (set `MLFLOW_TRACKING_URI`)
- VS Code config: `.vscode/mcp.json` (uses `servers` key, not `mcpServers`)
- No `.mcp.json` for Claude Code — MCP is wired in some other way here.

## context7 Library IDs

- `/marimo-team/marimo` - marimo docs
- `/mlflow/mlflow` - mlflow docs
- `/mathlab/pina` - PINA docs

## Cross-Platform Hooks

Pattern: `bash -c '...' 2>/dev/null || powershell ...`

## Code Style

- Use `uv` workflow (uv add, uv sync, uv tool install)
- No god classes - split into smaller modules
- No unnecessary wrappers - use core library features directly
- Skills in `.claude/Skills/` are git-tracked

## Composition-first PDEs (Phase A-0 + B-F, 2026-04-24)

No hardcoded `ProblemKind`. Agents compose PDEs from primitives:

- `services/composer.py::compose_problem(ProblemSpec)` — compiles
  a `pina.Problem` subclass at runtime via `sympy.lambdify` +
  `pina.operator.grad/laplacian`. Auto-detects forward vs inverse
  by residual-signature length (2 vs 3 args).
- `services/mesh_domain.py::MeshDomain` — PINA `DomainInterface`
  adapter over `meshio`; barycentric sampling per cell kind.
- `services/design.py::ConstraintAggregator` — penalty +
  augmented-Lagrangian handler for PDE-constrained optimisation.
- `src/marimo_flow/control/` — rolling-horizon scipy SLSQP MPC
  (`run_mpc_step`, `simulate_closed_loop`). No casadi / do-mpc.
- `core/viz3d.py` — plotly `Mesh3d`/`Volume`/`Scatter3d`/`Isosurface`
  for 3D domain + field visualisation. **No pyvista** (avoid 150 MB
  VTK stack).

Demo notebooks: `examples/03_navier_stokes_3d_cavity.py` (3D NS via
composer), `examples/04_mpc_heat_rod.py` (closed-loop MPC on a heat
surrogate).

## Agents (`src/marimo_flow/agents/`)

Multi-agent PINA team on `pydantic-graph` with 9 nodes:
`TriageNode` (start) → `RouteNode` (dispatcher) → `ProblemNode`, `ModelNode`,
`SolverNode`, `TrainingNode`, `ValidationNode`, `MLflowNode`, `NotebookNode`.

Layout (SPEC-driven):
- `schemas/` — typed Pydantic models for every handoff artefact:
  `TaskSpec`, `ProblemSpec` (composition-first), `EquationSpec`,
  `SubdomainSpec`, `ConditionSpec`, `DerivativeSpec`, `ObservationSpec`,
  `UnknownParameterSpec`, `MeshSpec`, `NoiseSpec`, `OptimizationPlan`,
  `DesignVariableSpec`, `ConstraintSpec`, `ControlPlan`,
  `ControlVariableSpec`, `StateSpec`, `ModelSpec`, `SolverPlan`,
  `RunConfig`, `DatasetBinding`, `ArtifactRef`, `AgentDecision`,
  `HandoffRecord`, `ValidationReport`, `ExperimentRecord`.
  Only `ModelKind` and `SolverKind` remain `Literal`s (finite PINA
  built-ins); problem physics is free-form via `EquationSpec`.
- `toolsets/` — `problem`, `model`, `solver`, `training`, `validation`,
  `data`, `design`, `control`, `curator`, `skills`, `lead`. Each is
  a `FunctionToolset[FlowDeps]` singleton.
- `services/` — `ProvenanceStore` (DuckDB, 16 tables), orchestrator
  policy helpers, experiment lifecycle, `composer`, `mesh_domain`,
  `design`, `preset_catalog`.
- `nodes/` — one module per graph node. `triage` builds a `TaskSpec` from
  free-form intent (fast-paths if one is already set). `validation` grades a
  training run and writes a `ValidationReport`. `route` emits `HandoffRecord`
  on every dispatch and short-circuits to `End` on escalate/reject verdicts.

Infrastructure:
- Orchestration: `pydantic-graph` builder API — `GraphBuilder` registers the
  v1 `BaseNode` subclasses as-is via `g.node(...)` (`BaseNode`/`End`/
  `GraphRunContext` survive into pydantic-graph v2; only the legacy `Graph`
  runner + its persistence machinery were deprecated). `build_graph()` returns
  a builder `Graph`; `agents.runner.run_graph()` drives it (`graph.run` /
  `graph.iter`) and logs FlowState snapshots to MLflow under `agent_state/`.
- Persistence + tracing: MLflow (`mlflow.pytorch.autolog()` +
  `mlflow.pydantic_ai.autolog()` on by default since mlflow >= 3.11.2 fixed
  the circular-ref crash; opt out with `MLFLOW_PYDANTIC_AI_AUTOLOG=0`)
- Provenance: DuckDB at `./provenance.duckdb` (configurable via
  `MARIMO_FLOW_PROVENANCE_DB` or `config.yaml`'s `provenance.db_path`).
  DuckDB 1.5.2 ships transitively via `marimo[sql]` — no extra project dep.
- LLMs: pydantic-ai `infer_model` with provider-prefixed specs; role→spec in
  `agents.deps.DEFAULT_MODELS` (10 roles: route, triage, notebook, problem,
  model, solver, training, validation, mlflow, lead).
- Each sub-agent loads its skill from `.claude/Skills/<name>/SKILL.md` via
  `build_skill_instructions()` — lazy, no message-history bloat.
- Lead agent (`build_lead_agent`) exposed three ways: `mo.ui.chat`, an A2A
  server (`fasta2a.pydantic_ai.agent_to_a2a`), and an AG-UI server (bare
  Starlette + `AGUIAdapter.dispatch_request`). `run_pina_workflow` wraps every
  graph run in an `ExperimentRecord` (running → completed / failed).
- `FlowState` holds MLflow URIs **and** the typed specs; live
  PINA/torch objects live in `FlowDeps.registry` keyed by URI.
  `FlowState.to_jsonable()` renders Pydantic fields via
  `model_dump(mode="json")` so snapshots round-trip through JSON.

Testing:
- `tests/agents/test_*.py` — 225 passing, 0 xfailed (baseline 2026-06-10).
- `test_demos_compose.py` smoke-tests the notebook spec paths so
  schema drift breaks the test suite, not the user-facing demos.
- Nodes that use MCP toolsets (Notebook, MLflow): stub `build_*_mcp` with
  `pydantic_ai.toolsets.FunctionToolset()` to avoid live server connections,
  and pin `TestModel(call_tools=[])`.
- Toolset unit tests: call `toolset.tools["name"].function(ctx, ...)` directly
  with a plain `ctx` wrapper and `FlowDeps(provenance_db_path=":memory:")`
  — no graph, no LLM.
- Spec-setting in the build-* toolsets is wrapped in `contextlib.suppress`
  so TestModel's dummy kinds (e.g. `"a"`) don't break the MLflow-only path.
- Use `provenance_db_path=":memory:"` in tests to avoid leaving DB files.
