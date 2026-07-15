# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-15

### Added
- **Parametric heat-rod control feature** (`marimo_flow.control.heat_rod`, `control.pinn_surrogate`): explicit FTCS ground-truth plant (`FiniteDifferenceHeatRod`), a parametric PINN problem builder (`build_heat_rod_problem_spec` — `T(x,t,u)` with the right-boundary temperature `T(1,t,u)=u` as the control input), an MLP reduced-order surrogate (`train_step_surrogate`), and a PINN field-mean → scalar rollout surrogate (`make_pinn_rollout_surrogate` / `register_pinn_surrogate`) for MPC horizon planning.
- **Composition-first control inputs**: `ControlParameterSpec` + `ProblemSpec.control_parameters` (exogenous MPC inputs merged into `domain_bounds` and sampled during PINN training) and a `parametric_dirichlet` `ConditionSpec.kind` that pins an output field to a sampled input axis (`output_field = u`).
- `register_pinn_surrogate_tool` on the control toolset — wraps a trained PINN solver as an MPC surrogate in `deps.registry`.

### Changed
- **Dependencies refreshed** via `uv lock --upgrade`: pydantic-ai-slim 1.84 → 2.10, torch 2.11 → 2.13, mlflow 3.11 → 3.13, transformers 5.5 → 5.13, PINA 0.2.6 → 0.3.1, marimo 0.23.1 → 0.23.9, plus transitive bumps.
- **Migrated to the PINA 0.3 API**: `DomainInterface` moved to `pina.domain`, `FixedValue` to `pina.equation.zoo`, `AbstractProblem` → `BaseProblem`, the PINN-family solvers aliased to their renamed `*SingleModelSolver` forms, and `discretise_domain()` no longer accepts `domains="all"` (the default now samples all domains). `pytest` is warning-free again.
- **pydantic-ai upgraded to 2.x**: the bundled `a2a` extra was dropped, so `fasta2a` is now an explicit direct dependency (the `agent_to_a2a` API is unchanged); the `pydantic-ai-slim` floor is bumped to `>=2.10.0` and the `[ag-ui]` extra retained.
- **Agent graph migrated to the builder-based `pydantic-graph` API**: `build_graph()` now uses `GraphBuilder`, registering the nine v1 `BaseNode` nodes as-is via `g.node(...)`. New `agents.runner.run_graph` drives the run (`graph.run` / `graph.iter`) and logs `FlowState` snapshots to MLflow under `agent_state/`, replacing the deprecated `BaseNode`-`Graph` runner + its persistence machinery.
- MCP builders migrated to `MCPToolset` (from deprecated `MCPServerStreamableHTTP` / `MCPServerStdio`); transport assertions now probe `toolset.client.transport`.
- A2A / AG-UI servers migrated off the deprecated `Agent.to_a2a()` / `Agent.to_ag_ui()`: A2A uses `fasta2a.pydantic_ai.agent_to_a2a`, AG-UI is a bare Starlette app dispatching to `AGUIAdapter.dispatch_request` (the previously-xfailed AG-UI ASGI test now passes).
- `mlflow.pydantic_ai.autolog()` is on by default — mlflow ≥ 3.11.2 fixed the circular-reference crash (mlflow#22693), now falling back to a repr dump. Opt out with `MLFLOW_PYDANTIC_AI_AUTOLOG=0`.
- README reframed as an agentic scientific-computing platform; `pyproject.toml` description + keywords and the GitHub repo About/topics updated to match.
- README gained an architecture diagram (natural language → pydantic-ai → PINA / MLflow / marimo), a consolidated stack table (pydantic-ai, PINA + PyTorch, MLflow, DuckDB, Polars, Plotly, marimo, MCP, Docker CPU/CUDA/XPU), and a **Claude Code integration** section covering the marimo MCP server + the marimo-pair live-kernel workflow.
- Cleared pre-existing ruff findings surfaced by the ruff bump (typer `B008` via `extend-immutable-calls`, `SIM300`, `B017`); `ruff check` is clean again.

### Removed
- `MLflowStatePersistence` (`pydantic-graph` persistence backend) — the builder runner exposes no persistence hook; FlowState snapshotting moved into `agents.runner`.

## [0.3.1] - 2026-04-28

### Added
- **Composition-first PDE architecture** (`services/composer.py::compose_problem`) — agents emit typed `EquationSpec` + `SubdomainSpec` + `ConditionSpec` and the composer compiles a `pina.Problem` subclass at runtime via `sympy.lambdify` + `pina.operator.grad/laplacian`. No hardcoded `ProblemKind` enum.
- **Phase B-F feature set**:
  - Inverse problems via `UnknownParameterSpec` + `ObservationSpec` (data-fitting → `pina.LearnableParameter` + 3-arg residual).
  - Mesh geometry via `MeshSpec` + `meshio` + `services/mesh_domain.py::MeshDomain` (barycentric sampling for tri/tetra/quad/hex).
  - Design optimisation via `OptimizationPlan` + `DesignVariableSpec` + `ConstraintSpec` (Optuna TPE / scipy SLSQP with penalty + augmented-Lagrangian handling in `services/design.py`).
  - Stochastic / non-local residuals via `NoiseSpec` (white / colored / fbm); fractional Laplacian via Riesz-kernel Monte-Carlo quadrature.
  - **MPC**: new `marimo_flow.control` package — rolling-horizon scipy SLSQP (`run_mpc_step`, `simulate_closed_loop`) on a trained PINN surrogate.
- `TriageNode` (start node) — parses free-form intent into a typed `TaskSpec`.
- `ValidationNode` — grades runs against `task_spec.constraints` and emits a `ValidationReport` with `accept/retry/escalate/reject` verdict; `RouteNode` short-circuits to `End` on `escalate/reject` (HITL).
- `services/provenance.py` — DuckDB `ProvenanceStore` with 13 tables (tasks, experiments, agent decisions, validation verdicts, handoffs, lineage, artifacts, …). DuckDB 1.5.2 ships transitively via `marimo[sql]`.
- `core/viz3d.py` — 3D plotly visualisation (`Mesh3d`, `Volume`, `Scatter3d`, `Isosurface`). No pyvista / VTK dependency.
- New demo notebooks: `02_provenance_dashboard.py` (DuckDB review surface), `03_navier_stokes_3d_cavity.py` (3D NS via composer), `04_mpc_heat_rod.py` (closed-loop MPC on a PINN surrogate).
- `marimo-flow` CLI entry point (`marimo_flow.cli:main`) with retries + safety-suppression around spec-setting.
- Provider-agnostic model config — `config.yaml` (see `config.yaml.example`) and `MARIMO_FLOW_MODEL_<ROLE>` env vars; any pydantic-ai provider works (openai, anthropic, groq, mistral, google-gla, bedrock, together, fireworks, openrouter, deepseek, cerebras, xai, ollama, huggingface, …).
- E2E regression test for Windows NTFS snapshot-id sanitization (`tests/agents/test_persistence_*`).

### Changed
- Multi-agent team now **drives PINA end-to-end** via the toolset layer instead of merely answering questions; nine graph nodes (`TriageNode` → `RouteNode` → `Problem`/`Model`/`Solver`/`Training`/`Validation`/`MLflow`/`Notebook`).
- MLflow local default pinned to `data/mlflow/{db,artifacts}/`; `./mlruns/` no longer used.
- `Dockerfile.xpu`: switched base from archived `intel/intel-extension-for-pytorch:2.8.10-xpu` (IPEX archived 2026-03-30) to `ubuntu:22.04` + Intel GPU runtime (`libze-intel-gpu1`, `libze1`, `intel-opencl-icd`); torch/torchvision/torchaudio/triton-xpu installed from the official PyTorch XPU index per Intel's "use PyTorch directly" guidance.
- `Dockerfile` (CPU): bumped PyG wheel URL from `torch-2.10.0+cpu` to `torch-2.11.0+cpu` to match the locked torch version.
- `uv` base image bumped to `ghcr.io/astral-sh/uv:0.11.7` across all three Dockerfiles.
- Documentation: `docs/INDEX.md` expanded; `scripts/` bash helpers dropped, docs reference direct `uv` commands; repo root cleaned, `.gitignore` hardened against MLflow stragglers.

### Fixed
- **Persistence on Windows**: snapshot IDs are now sanitized before being used as filenames (NTFS forbids `:`); previously `MLflowStatePersistence` could fail to write graph snapshots on Windows hosts.
- `python-publish.yml` now dispatches `docker-publish.yml` after creating the GitHub Release; previously the release was created via `GITHUB_TOKEN`, which by GitHub policy does not trigger downstream workflows, so Docker images were not built automatically.
- `python-publish.yml` `github-release` job now has `actions: write` permission (required by `gh workflow run docker-publish.yml`); the previous job-level `permissions:` block silently dropped it.
- `docker-publish.yml` removed dead Docker Hub login (workflow only pushes to GHCR); the `if: ${{ secrets.DOCKERHUB_USERNAME != '' }}` pattern was rejected by GitHub's workflow validator.
- `Dockerfile.xpu`: defensive `getent` guard around `groupadd` / `useradd` (mirrors `Dockerfile.cuda`); ubuntu:22.04 currently has no UID 1000, but base images change.
- Examples: `marimo export html` revealed that `02_provenance_dashboard.py`, `03_navier_stokes_3d_cavity.py`, `04_mpc_heat_rod.py` violated marimo's single-definition constraint (multiple cells defining `pd`/`rows`/`df`/`fig`/`np`) and `02` had inner returns that tripped marimo's static analyser. Variables now cell-unique, pandas dropped from `02` (mo.ui.table accepts list-of-dicts, ProvenanceStore.query returns dicts), `np` shared via cell-argument in `04`, single-return-point in `_problem_3d_view`.
- `core.train_solver`: pin Lightning's `default_root_dir` so `checkpoints/` and `lightning_logs/` no longer leak into the working directory. When an MLflow run is active and uses a local file-store, point Lightning at the run's artifact directory — checkpoints land *inside* the run and show up in the MLflow artifacts tab. Otherwise fall back to `data/mlflow/lightning/` next to the MLflow backend store. Override via the new `default_root_dir=` parameter.
- All 26 pytest warnings cleaned — 216 passing, 0 warnings (mlflow filesystem-backend FutureWarning, PINA Lightning val-dataloader noise, torch_geometric self-deprecation, etc., either fixed at the call site or pinned in `[tool.pytest.ini_options].filterwarnings`).

### Removed
- `transformers[sentencepiece]` extra (unused, triggered SWIG `DeprecationWarning`).
- Dead `Black` dev dependency (replaced by `ruff format`).
- Phantom notebook references in docs (notebooks that were renamed/removed during the Phase B-F restructuring).
- `scripts/` bash helper directory — docs now reference direct `uv` commands instead.

## [0.3.0] - 2026-04-22

### Added
- Multi-agent PINA team (`marimo_flow.agents`) built on `pydantic-graph` + MLflow.
- `RouteNode` classifier dispatching to `Notebook`, `Problem`, `Model`, `Solver`, `MLflow` sub-nodes.
- `FlowState` (JSON-serialisable) + `FlowDeps` (in-memory registry) — non-serialisable PINA/torch objects live in `FlowDeps.registry` keyed by MLflow artifact URIs.
- `MLflowStatePersistence` — `pydantic-graph` persistence backend logging snapshots as MLflow artifacts.
- Skill loader (`marimo_flow.agents.skills`) — agents load `.claude/Skills/<name>/SKILL.md` as lazy `instructions=` callables; supports concatenating multiple skills per role.
- `_define_problem` / `_define_model` / `_define_solver` — open-form spec tools (no fixed enum); the agent designs the spec to fit the problem.
- Lead agent (`build_lead_agent`) wraps the graph as one tool; exposed via marimo chat (`lead_chat`), A2A (`server.a2a`), and AG-UI (`server.ag_ui`).
- A2A AgentCard with one `Skill` per sub-node role for capability discovery by external agents.
- Ollama-Cloud `OpenAIChatModel` factory (`get_model`) — single endpoint for local + cloud `:cloud` models, no separate proxy.
- `examples/lab.py` rewritten as full PINA team chat demo with state inspector and live mermaid diagram.
- `CITATION.cff` for Zenodo DOI integration on GitHub Releases.

### Changed
- `pydantic-ai-slim` upgraded to `[a2a, ag-ui, openai]` extras for protocol support.
- `OpenAIModel` → `OpenAIChatModel` (deprecation in pydantic-ai 1.84).
- `pyproject.toml` description + keywords updated to reflect PINA, agents, MCP, and Ollama integration.
- README: dropped marketing claims (`Production Ready` subsection, etc.), removed obsolete `docs/RESEARCH_SUMMARY.md`.

### Fixed
- `_define_*` helpers now use `MlflowClient` with explicit `state.mlflow_run_id` to avoid silent artifact misroute when no module-level active run exists (e.g. inside `await graph.run(...)`).

## [0.2.0] - 2026-03-26

### Added
- Multi-platform Docker images (CPU, CUDA, XPU) published to GHCR
- PINA integration: ProblemManager, ModelFactory, SolverManager, WalrusAdapter, Optuna visualization helpers
- MCP integration: marimo, mlflow, context7 servers pre-configured
- Claude Code skills for marimo, mlflow, and pina
- Tag-based PyPI publish workflow with auto GitHub Release
- Security: pinned minimum versions for all Dependabot-flagged transitive deps (authlib, pillow, cryptography)
- Dependabot auto-merge workflow

### Changed
- Examples reduced to 3 focused notebooks (MLflow Console, PINA Walrus Solver, PINA Live Monitoring)
- Dependencies simplified to 3 core (marimo, mlflow, pina-mathlab) + optional `[all]` extra for torch/torch-geometric

### Removed
- Redundant tutorials and snippets module
- CONTRIBUTING.md and docs/ reference documentation

## [0.1.3] - 2025-11-23

### Added
- Complete example progression (00-08) covering ML lifecycle
- Example 07: LoRA fine-tuning for Large Language Models
- Example 08: Graph Neural Networks with PyTorch Geometric
- New snippets: agent.py, duckdb_sql.py, openvino_1.py, rag.py
- Tools directory with ollama_manager.py and openvino_manager.py
- Comprehensive reference documentation in docs/ directory:
  - marimo-quickstart.md
  - polars-quickstart.md
  - plotly-quickstart.md
  - pina-quickstart.md
  - integration-patterns.md
- Docker helper scripts (marimo-flow-agent, marimo-flow-code)
- .marimo.toml configuration for runtime settings

### Changed
- Reorganized examples with progressive numbering (00-08)
- Restructured project layout for better clarity
- Updated README.md with complete project structure
- Enhanced .gitignore for user-specific configs and cache files
- Improved Docker configuration

### Removed
- experimental/ directory (split into examples/ and tools/)
- Domain-specific apps (astrophotography, cosmos analysis)
- Redundant PyTorch and snippet files
- configs/ empty directory
- Old example files with inconsistent naming

### Fixed
- Git repository structure consistency
- File organization in root directory
- Documentation alignment with actual structure

## [0.1.2] - 2025-10-18

### Added
- PINA (Physics-Informed Neural Networks) integration
- PyTorch Geometric snippets for graph neural networks
- Enhanced Dockerfile with PyG dependencies

### Changed
- Updated dependencies to latest versions
- Improved Docker build process

## [0.1.1] - 2025-07-14

### Added
- Docker configuration with docker-compose
- Custom CSS for Marimo UI styling
- GitHub Pages workflow (Jekyll)
- Python publish workflow for PyPI

### Changed
- Updated Python version requirement to 3.11+
- Cleaned up configuration files
- Refactored Marimo Flow project structure

### Fixed
- Configuration file consistency
- Project metadata alignment

## [0.1.0] - 2025-07-08

### Added
- Initial release of Marimo Flow
- Core ML pipeline examples (00-06)
- MLflow integration for experiment tracking
- Marimo reactive notebooks
- Basic snippets for common patterns:
  - mlflow_setup.py
  - interactive_params.py
  - data_loading.py
  - altair_visualization.py
- Docker support with docker-compose
- SQLite backend for MLflow
- Progressive example structure
- MIT License
- Basic README.md documentation

### Features
- Reactive notebook development with Marimo
- Seamless MLflow experiment tracking
- Interactive parameter tuning
- Model registry and versioning
- Production pipeline examples
- Git-friendly .py notebooks
- Docker one-command setup

---

## Version History Summary

- **0.4.0** (2026-07-15) - Parametric heat-rod control + MPC PINN surrogate, PINA 0.3 / pydantic-ai 2.x migration, README architecture + stack + Claude Code integration
- **0.3.1** (2026-04-28) - Composition-first PDEs, Phases B-F (inverse, mesh, design, stochastic, MPC), Triage/Validation nodes, DuckDB provenance, plotly 3D viz, Windows persistence fix
- **0.3.0** (2026-04-21) - Multi-agent PINA team (`pydantic-graph` + MLflow + Ollama Cloud), CITATION.cff
- **0.2.0** (2026-03-26) - Multi-platform Docker, PINA integration, MCP servers, simplified deps
- **0.1.3** (2025-11-23) - Major restructuring, advanced examples, comprehensive docs
- **0.1.2** (2025-10-18) - PINA and PyG integration
- **0.1.1** (2025-07-14) - Docker and CI/CD improvements
- **0.1.0** (2025-07-08) - Initial release

[Unreleased]: https://github.com/synapticore-io/marimo-flow/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/synapticore-io/marimo-flow/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/synapticore-io/marimo-flow/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/synapticore-io/marimo-flow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/synapticore-io/marimo-flow/releases/tag/v0.1.0
