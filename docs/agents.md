# PINA Multi-Agent Team

`marimo_flow.agents` is a reactive multi-agent team that orchestrates
PINA workflows end-to-end. Built on `pydantic-graph`, traced + persisted
through MLflow, and exposed via marimo's chat UI plus optional A2A and
AG-UI ASGI servers.

The team **drives PINA** — it doesn't just answer questions. Free-form
intent → typed `TaskSpec` → composed `ProblemSpec`/`ModelSpec`/
`SolverPlan` → trained surrogate → validation verdict, all logged.

## Quick start

```python
from marimo_flow.agents import lead_chat, FlowDeps
import marimo as mo

deps = FlowDeps()  # uses sqlite:///data/mlflow/db/mlflow.db by default
chat = mo.ui.chat(lead_chat(deps=deps))
chat
```

See [`examples/lab.py`](../examples/lab.py) for the full demo notebook
(state inspector, live mermaid diagram of the graph).

## Graph topology — 9 nodes

```
TriageNode → RouteNode ─┬─ ProblemNode
                        ├─ ModelNode
                        ├─ SolverNode
                        ├─ TrainingNode
                        ├─ ValidationNode → (accept/retry/escalate/reject)
                        ├─ MLflowNode
                        └─ NotebookNode
```

- **`TriageNode`** (start) — parses free-form user intent into a typed
  `TaskSpec`.
- **`RouteNode`** — dispatcher. Emits a `HandoffRecord` on every
  dispatch and short-circuits to `End` when the validation verdict is
  `escalate` / `reject` (SPEC §13 HITL).
- **`ValidationNode`** — grades the run against `task_spec.constraints`
  and records a `ValidationReport` with an `accept/retry/escalate/reject`
  verdict.

## Roles — one skill per agent

Each role loads its `.claude/Skills/<name>/SKILL.md` as `instructions=`
where applicable:

| Role | Purpose | Skill(s) |
|---|---|---|
| `triage` | parses free-form intent into a `TaskSpec` | — |
| `notebook` | marimo MCP cell ops | `marimo`, `marimo-pair` |
| `problem` | defines a PINA Problem from an open spec | `pina-problem` |
| `model` | designs a neural architecture | `pina-model` |
| `solver` | wires Solver + Trainer config | `pina-solver` |
| `training` | runs `pina.Trainer.fit()` | `pina-training` |
| `validation` | grades the run against constraints | — |
| `mlflow` | MLflow MCP tracking + registry | `mlflow` |
| `lead` | chat / A2A / AG-UI front-end; wraps the graph | — |

## Typed specs + provenance (SPEC §8, §12)

Every graph run builds typed `ProblemSpec` / `ModelSpec` / `SolverPlan`
/ `RunConfig` on `FlowState` and mirrors them — plus `AgentDecision`,
`HandoffRecord`, `ValidationReport`, `ExperimentRecord`, `ArtifactRef`,
and lineage edges — into a DuckDB provenance store
(`./provenance.duckdb` by default, or `MARIMO_FLOW_PROVENANCE_DB`).

MLflow still owns the binary artifacts; DuckDB owns the queryable index.
DuckDB 1.5.2 ships transitively via `marimo[sql]` — no extra project dep.

```python
from marimo_flow.agents.services import ProvenanceStore

store = ProvenanceStore("provenance.duckdb")
print(store.query(
    "SELECT title, verdict FROM tasks t "
    "LEFT JOIN validation_reports v USING (task_id) "
    "ORDER BY t.created_at DESC LIMIT 10"
))
```

See [`examples/02_provenance_dashboard.py`](../examples/02_provenance_dashboard.py)
for a marimo review surface over the full DuckDB schema (16 tables).

## Models — provider-agnostic

Provider-prefixed specs (`"<provider>:<model>"`) resolved through
pydantic-ai's `infer_model`. Defaults in
[`marimo_flow.agents.deps.DEFAULT_MODELS`](../src/marimo_flow/agents/deps.py)
all point at Ollama Cloud (`http://localhost:11434/v1`,
`:cloud`-suffixed tags).

Override per role either via `config.yaml` at the repo root (see
`config.yaml.example`) or with `MARIMO_FLOW_MODEL_<ROLE>=<spec>` env
vars. Any provider in the pydantic-ai catalogue works:

> openai · anthropic · groq · mistral · google-gla · bedrock · together
> · fireworks · openrouter · deepseek · cerebras · xai · ollama ·
> huggingface · …

## Standalone servers

The lead agent (`build_lead_agent`) is exposed three ways:

```bash
# In-notebook chat (default)
# uses mo.ui.chat(lead_chat(deps=deps)) — see examples/lab.py

# A2A protocol (Anthropic Agent-to-Agent)
uv run python -m marimo_flow.agents.server.a2a    # :8000

# AG-UI protocol (CopilotKit's agent UI bridge)
uv run python -m marimo_flow.agents.server.ag_ui  # :8001
```

The A2A `AgentCard` exposes one `Skill` per sub-node role for capability
discovery by external agents.

## Configuration

```yaml
# config.yaml at repo root
provenance:
  db_path: ./provenance.duckdb   # or MARIMO_FLOW_PROVENANCE_DB

models:
  triage:     ollama:gpt-oss:20b-cloud
  problem:    ollama:gpt-oss:20b-cloud
  validation: anthropic:claude-sonnet-4-6
  # … one entry per role; missing roles fall back to DEFAULT_MODELS
```

See [`config.yaml.example`](../config.yaml.example) for the full role
list and provider syntax.

## Implementation layout

```
src/marimo_flow/agents/
├── nodes/        — one module per graph node
├── schemas/      — typed Pydantic models for every handoff artefact
├── toolsets/     — FunctionToolset[FlowDeps] per role
├── services/     — ProvenanceStore, composer, mesh_domain, design …
├── server/       — A2A + AG-UI ASGI app builders
├── deps.py       — FlowDeps (in-memory registry) + DEFAULT_MODELS
└── lead.py       — graph wiring + ExperimentRecord lifecycle
```

## See also

- [`examples/lab.py`](../examples/lab.py) — multi-agent team chat demo
- [`examples/02_provenance_dashboard.py`](../examples/02_provenance_dashboard.py) — DuckDB review surface
- [`docs/pydantic-ai-toolsets-reference.md`](pydantic-ai-toolsets-reference.md) — per-role toolset API
- [`docs/roadmap.md`](roadmap.md) — Phase A-0 → F status
- [`docs/mcp-setup.md`](mcp-setup.md) — MCP server configuration
