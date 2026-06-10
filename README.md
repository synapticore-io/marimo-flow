# Marimo Flow 🌊


[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Marimo](https://img.shields.io/badge/Marimo-Latest-orange?logo=python&logoColor=white)](https://marimo.io)
[![MLflow](https://img.shields.io/badge/MLflow-Latest-blue?logo=mlflow&logoColor=white)](https://mlflow.org)
[![MCP](https://img.shields.io/badge/MCP-Enabled-green?logo=anthropic&logoColor=white)](https://docs.marimo.io/guides/editor_features/mcp/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![Version](https://img.shields.io/badge/Version-0.3.1-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/synapticore-io/marimo-flow)
[![Contributing](https://img.shields.io/badge/Contributing-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

*Like marimo algae drifting in crystal waters, your data flows and evolves – each cell a living sphere of computation, gently touching others, creating ripples of reactive change. In this digital ocean, data streams like currents, models grow like organic formations, and insights emerge naturally from the depths. Let your ML experiments flow freely, tracked and nurtured, as nature intended.*



<div align="center">

https://github.com/user-attachments/assets/3bc24463-ff42-44a7-ae61-5d500d29688c



</div>


## What is marimo-flow? 🚀

**An agentic scientific-computing platform for physics-informed ML.**
A reactive multi-agent team orchestrates PINA / PINN workflows
end-to-end over a `pydantic-graph` state machine — backed by MLflow for
tracing and persistence, and exposed through marimo's chat UI plus
optional A2A and AG-UI ASGI servers.

Describe a PDE in natural language; the team composes the Problem,
designs the network, wires the solver, trains it, and grades the run —
every handoff typed, logged to MLflow, and indexed in a DuckDB
provenance store. A classic `marimo_flow.core` API is still there for
when you'd rather drive PINA by hand. Composition-first throughout: no
hardcoded PDE factories, plus a Docker deployment story for CPU /
NVIDIA / Intel GPUs.

## 🧠 PINA — composition-first (no hardcoded PDE factories)

The core idea: **agents emit typed `EquationSpec` + `SubdomainSpec` +
`ConditionSpec` and the composer compiles a `pina.Problem` subclass at
runtime** via `sympy.lambdify` + `pina.operator.grad/laplacian`. There
is no `ProblemKind` enum — any PDE sympy can express is reachable.

| Capability | What it gives you | Spec(s) |
|---|---|---|
| **Composition-first PDEs** | `compose_problem(ProblemSpec)` builds the Problem class on demand. | `EquationSpec`, `SubdomainSpec`, `ConditionSpec`, `DerivativeSpec` |
| **Inverse problems** | data-fitting → `pina.LearnableParameter`, 3-arg residual auto-routed. | `UnknownParameterSpec`, `ObservationSpec` |
| **Mesh geometry** | unstructured STL/OBJ/VTK/GMSH meshes as the spatial domain. Barycentric sampling per cell kind (tri/tetra/quad/hex). | `MeshSpec` + `services/mesh_domain.py` |
| **3D visualisation** | plotly `Mesh3d` / `Volume` / `Scatter3d` / `Isosurface`. **No** pyvista / 150 MB VTK stack. | `core/viz3d.py` |
| **Design optimisation** | Optuna TPE / scipy SLSQP with penalty + augmented-Lagrangian handling. | `OptimizationPlan`, `DesignVariableSpec`, `ConstraintSpec` |
| **Stochastic + non-local** | white / colored / fbm noise; fractional Laplacian via Riesz-kernel Monte-Carlo quadrature. | `NoiseSpec` |
| **MPC** | rolling-horizon scipy SLSQP on a trained PINN surrogate. | `marimo_flow.control` package |
| **Walrus foundation model** | adapter for Poisson-class problems. | `core.FoundationModelAdapter` |

Code lives in [`src/marimo_flow/core/`](src/marimo_flow/core/) (PINA
solvers + training + viz3d), [`src/marimo_flow/control/`](src/marimo_flow/control/)
(MPC), and [`src/marimo_flow/agents/services/`](src/marimo_flow/agents/services/)
(composer, mesh-domain, design aggregator).

## Features ✨

- **🧑‍🚀 Multi-agent PINA team** — free-form intent → typed `TaskSpec`
  → composed `ProblemSpec` / `ModelSpec` / `SolverPlan` → trained
  surrogate → validation verdict, over a 9-node `pydantic-graph` state
  machine (see [docs/agents.md](docs/agents.md)).
- **🔌 Three transports** — in-notebook `mo.ui.chat`, an A2A ASGI
  server, and an AG-UI ASGI server, all wrapping the same lead agent.
- **🧩 Provider-agnostic LLMs** — per-role model specs resolved through
  pydantic-ai's `infer_model`; defaults target Ollama Cloud, any
  provider in the catalogue works via `config.yaml` / env vars.
- **🔬 MLflow tracing + DuckDB provenance** — every run, model, and
  metric tracked under `data/mlflow/{db,artifacts}/`; typed specs,
  decisions, handoffs, and lineage mirrored into a 16-table DuckDB
  store. Lightning checkpoints land *inside* the active run.
- **🧠 Composition-first PINA** — agents compose PDEs from primitives;
  no hardcoded `ProblemKind` (see below).
- **📓 Reactive notebooks** — Git-friendly `.py` notebooks with
  automatic dependency tracking.
- **🤖 MCP-powered AI** — `marimo`, `mlflow`, and `context7` MCP
  servers wired up; live library docs without leaving your notebook.
- **🐳 Multi-platform Docker** — CPU, CUDA, Intel XPU images on GHCR.

## Quick Start 🏃‍♂️

### Talk to the agent team

The fastest path is the in-notebook chat. The lead agent dispatches to
the specialist team and streams its answer back:

```python
from marimo_flow.agents import lead_chat, FlowDeps
import marimo as mo

deps = FlowDeps()  # resolves per-role models + MLflow URI from config / .env / env vars
chat = mo.ui.chat(
    lead_chat(deps=deps),
    prompts=["Solve a 1D Poisson equation on [0,1] with u(0)=u(1)=0 using a PINN."],
)
chat
```

Defaults target Ollama Cloud (`:cloud` tags via a local Ollama
endpoint). Copy [`config.yaml.example`](config.yaml.example) to
`config.yaml` to point any role at OpenAI, Anthropic, Groq, … — auth
comes from each provider's standard env var.

Prefer a terminal? The CLI runs the same graph:

```bash
uv run marimo-flow solve "Solve the Burgers equation with a small PINN"
uv run marimo-flow solve -m lead=anthropic:claude-sonnet-4-6 "..."   # override a role
uv run marimo-flow config-show        # print the resolved models + URIs
uv run marimo-flow lab                # open examples/lab.py in marimo
```

Or expose the team as a standalone ASGI server:

```bash
uv run python -m marimo_flow.agents.server.a2a     # A2A protocol   → :8000
uv run python -m marimo_flow.agents.server.ag_ui   # AG-UI protocol → :8001
```

### With Docker (Recommended)

```bash
git clone https://github.com/synapticore-io/marimo-flow.git
cd marimo-flow
docker compose -f docker/docker-compose.yaml up --build -d

# Marimo:  http://localhost:2718
# MLflow:  http://localhost:5000
```

#### Image Variants

| Variant | Image Tag | Use Case |
|---|---|---|
| **CPU** | `ghcr.io/synapticore-io/marimo-flow:latest` | No GPU (lightweight) |
| **CUDA** | `ghcr.io/synapticore-io/marimo-flow:cuda` | NVIDIA GPUs |
| **XPU** | `ghcr.io/synapticore-io/marimo-flow:xpu` | Intel Arc / Data Center GPUs |

GPU compose files: `docker-compose.cuda.yaml` (requires nvidia-docker)
and `docker-compose.xpu.yaml` (requires Intel GPU drivers).

### Local Development

→ See **[SETUP.md](SETUP.md)** — bare-metal `uv` path, MLflow + marimo
processes, MCP setup table, troubleshooting. Five minutes from clone
to running notebook.

## Example Notebooks 📚

All notebooks live in `examples/` and open with
`uv run marimo edit examples/<file>.py`.

| Notebook | What it does |
|---|---|
| **`01_pina_poisson_solver.py`** | Poisson PDE with baseline PINN or Walrus foundation model. MLflow + Optuna sweep analytics. Uses `marimo_flow.core` directly. |
| **`02_provenance_dashboard.py`** | DuckDB review surface over the agent provenance store. Five tables (tasks, experiments, decisions, validation, handoffs) + 3D preset preview. |
| **`03_navier_stokes_3d_cavity.py`** | 3D lid-driven cavity composed end-to-end from a declarative `ProblemSpec`. No hardcoded NS factory. Plotly mid-plane velocity slice. |
| **`04_mpc_heat_rod.py`** | Closed-loop MPC on a 1D heat-rod PINN surrogate. Trains the surrogate, then drives a rolling-horizon scipy-SLSQP MPC loop toward a temperature setpoint. |
| **`lab.py`** | Multi-agent PINA team chat demo (requires Ollama running locally). |

## Project Structure 📁

```
marimo-flow/
├── examples/                     # Demo notebooks
├── src/marimo_flow/
│   ├── core/                     # PINA solvers, training, plotly viz3d
│   ├── control/                  # Rolling-horizon MPC (scipy SLSQP)
│   └── agents/                   # Multi-agent team (pydantic-graph + MLflow)
│       ├── nodes/                # 9 graph nodes
│       ├── schemas/              # Typed Pydantic specs (ProblemSpec, …)
│       ├── toolsets/             # FunctionToolset per role
│       └── services/             # composer, mesh_domain, design,
│                                 #   provenance (DuckDB, 16 tables)
├── tests/                        # 226 passing, 1 xfailed
├── docker/                       # Dockerfiles + compose (CPU/CUDA/XPU)
├── docs/                         # Project documentation (see docs/INDEX.md)
└── data/mlflow/                  # MLflow storage (db + artifacts)
```

## Two Workflows

| Workflow | Import | Use Case |
|---|---|---|
| **Classic** (`core/`) | `from marimo_flow.core import ...` | You know the PDE, pick a solver, log to MLflow. See `examples/01_pina_poisson_solver.py`. |
| **Agents** (`agents/`) | `from marimo_flow.agents import lead_chat, FlowDeps` | Describe the problem in natural language; a multi-agent team composes Problem + Model + Solver. See `examples/lab.py`. |

Both write to the same MLflow backend (`data/mlflow/`). The two
packages do not depend on each other — pick whichever matches the task.

## PINA Multi-Agent Team 🧑‍🚀

The team **drives PINA workflows end-to-end** — it doesn't just answer
questions. Free-form intent → typed `TaskSpec` → composed `ProblemSpec`
→ trained surrogate → validation verdict, all logged to MLflow + a
DuckDB provenance store.

```
TriageNode → RouteNode ─┬─ ProblemNode
                        ├─ ModelNode
                        ├─ SolverNode
                        ├─ TrainingNode
                        ├─ ValidationNode → (accept/retry/escalate/reject)
                        ├─ MLflowNode
                        └─ NotebookNode
```

Nine graph nodes (`pydantic-graph`), one toolset + skill per role,
provider-agnostic LLM config, and three transport options (in-notebook
chat, A2A server, AG-UI server). See the [Quick Start](#talk-to-the-agent-team)
for the chat / CLI / server entry points.

→ Full architecture, role list, provenance schema, and config docs in
**[docs/agents.md](docs/agents.md)**.

## MCP Integration 🔌

marimo and AI-assisted IDEs share MCP servers for live documentation
and notebook operations. Pre-configured in `.marimo.toml` (in-notebook)
and `.vscode/mcp.json` (VS Code / Claude Code).

→ Full configuration reference in **[docs/mcp-setup.md](docs/mcp-setup.md)**.

## Documentation 📚

| File | What's in it |
|---|---|
| [SETUP.md](SETUP.md) | Bare-metal local-dev path (5 min, no Docker) |
| [docs/INDEX.md](docs/INDEX.md) | Index of project documentation |
| [docs/agents.md](docs/agents.md) | Multi-agent team architecture, roles, provenance schema |
| [docs/mcp-setup.md](docs/mcp-setup.md) | MCP server configuration across IDEs |
| [docs/pydantic-ai-toolsets-reference.md](docs/pydantic-ai-toolsets-reference.md) | Per-role agent toolset API |
| [docs/roadmap.md](docs/roadmap.md) | Phase A-0 → F status with file/test pointers |
| [CHANGELOG.md](CHANGELOG.md) | Release history (Keep a Changelog) |
| [CLAUDE.md](CLAUDE.md) | Guidance for AI agents working in this repo |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow, code style, test expectations |

## Contributing 🤝

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, code standards, and the PR process.

```bash
# Quick path
git checkout -b my-feature
uv run pytest                         # 226 passing
uv run ruff format . && uv run ruff check --fix .
# open a PR
```

## License 📄

MIT — see [LICENSE](LICENSE).

---

**Built with ❤️ using marimo, MLflow, and PINA.**
