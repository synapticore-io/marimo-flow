# MCP Setup Guide for marimo-flow

Complete guide for setting up Model Context Protocol (MCP) servers with marimo-flow across all development environments.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [MCP Servers](#mcp-servers)
- [Setup by Environment](#setup-by-environment)
  - [Local Development](#local-development)
  - [Claude Desktop](#claude-desktop)
  - [VSCode](#vscode)
  - [Cursor](#cursor)
- [MCP Tools Reference](#mcp-tools-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

marimo-flow provides **three MCP servers** for AI-powered development:

| MCP Server | Purpose | Tools |
|------------|---------|-------|
| **Marimo** | Notebook introspection & debugging | `get_active_notebooks`, `get_notebook_errors`, `get_cell_runtime_data`, `get_tables_and_variables` |
| **Context7** | Live documentation for Python libraries | `search_docs`, `get_library_docs` |
| **MLflow** | Experiment tracking & model management | `search_experiments`, `search_runs`, `log_metric`, `list_models` |

### Why MCP?

- 📚 **Live Documentation** - Access up-to-date library docs without leaving your workflow
- 🔍 **Notebook Introspection** - Debug notebooks by inspecting runtime state
- 📊 **Experiment Tracking** - Query and manage MLflow experiments via AI
- 🤖 **AI-First Development** - Context-aware assistance from Claude, Cursor, etc.

---

## Quick Start

### 1. Start Development Environment

Run each service in its own terminal:

```bash
# Terminal A — MLflow tracking + UI
uv run mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root data/mlflow/artifacts \
  --serve-artifacts

# Terminal B — Marimo with MCP
uv run marimo edit examples/ --mcp --no-token --headless --port 2718
```

For the full Docker stack instead, see [README → Quick Start with
Docker](../README.md#with-docker-recommended).

This starts:
- **MLflow Server**: http://localhost:5000
- **Marimo with MCP**: http://localhost:2718
- **Marimo MCP Endpoint**: http://localhost:2718/mcp/server

### 2. Verify Services

```bash
# Check MLflow
curl http://localhost:5000/health

# Check Marimo MCP
curl http://localhost:2718/mcp/server

# Check running services
ps aux | grep -E "mlflow|marimo"
```

### 3. Configure Your IDE

Choose your environment:
- [Claude Desktop](#claude-desktop) - Full MCP support
- [VSCode](#vscode) - Marimo extension + tasks
- [Cursor](#cursor) - AI coding with custom rules

---

## MCP Servers

### 1. Marimo MCP Server

**Purpose**: Introspect running notebooks, debug errors, analyze data

**How to start**:
```bash
marimo edit examples/ --mcp --port 2718
```

**Endpoint**: `http://localhost:2718/mcp/server`

**Available Tools**:
- `get_active_notebooks` - List all open notebooks
- `get_notebook_errors` - Find runtime errors in cells
- `get_cell_runtime_data` - Inspect cell execution state
- `get_tables_and_variables` - Analyze DataFrames and variables
- `get_database_tables` - Inspect DuckDB/SQL connections
- `get_lightweight_cell_map` - Get notebook structure
- `get_marimo_rules` - Get marimo best practices

**Example Usage** (with Claude):
```
User: "List active marimo notebooks"
Claude: [Uses get_active_notebooks]
→ Returns: ["01_pina_poisson_solver.py", "lab.py"]

User: "Find errors in the PINA solver notebook"
Claude: [Uses get_notebook_errors for 01_pina_poisson_solver.py]
→ Returns: Cell execution errors with stack traces

User: "Show me the variables in the Poisson solver notebook"
Claude: [Uses get_tables_and_variables]
→ Returns: Schema, row count, column types
```

---

### 2. Context7 MCP Server

**Purpose**: Live documentation for Python libraries

**How it works**: Automatically configured in `.marimo.toml` as a preset

**Endpoint**: `https://context7.com/api/v1/mcp/sse` (SSE transport)

**Available Tools**:
- `search_docs` - Search documentation for any Python library
- `get_library_docs` - Get comprehensive library reference

**Supported Libraries**: Polars, Pandas, NumPy, Plotly, Altair, Marimo, scikit-learn, PyTorch, and 1000+ more

**Example Usage**:
```
User: "How do I use polars window functions?"
Claude: [Uses search_docs for "polars window functions"]
→ Returns: Latest polars docs + code examples

User: "Show me plotly 3D scatter plot examples"
Claude: [Uses get_library_docs for plotly]
→ Returns: Current plotly API with working samples
```

---

### 3. MLflow MCP Server

**Purpose**: Query experiments, track metrics, manage models

**How to start**:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow mcp run
```

**Transport**: stdio (for Claude Desktop/Code)

**Available Tools**:
- `search_experiments` - Find ML experiments
- `get_experiment` - Get experiment details
- `search_runs` - Query training runs
- `get_run` - Get run details
- `log_metric` - Track experiment metrics
- `log_param` - Log hyperparameters
- `list_models` - Browse model registry
- `get_model_version` - Get model version info

**Example Usage**:
```
User: "What experiments are in my MLflow?"
Claude: [Uses search_experiments]
→ Returns: List of experiments with IDs and names

User: "Show me the best run for the PINA experiment"
Claude: [Uses search_runs with filter]
→ Returns: Run with highest accuracy metric

User: "Log a new metric to run abc123"
Claude: [Uses log_metric]
→ Logs: run_id=abc123, metric=accuracy, value=0.95
```

---

## Setup by Environment

### Local Development

#### Prerequisites
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

#### Configuration Files

**`.marimo.toml`** - MCP presets and server config:
```toml
[mcp]
presets = ["marimo", "context7"]

[mcp.mcpServers.mlflow]
command = "mlflow"
args = ["mcp", "run"]
env = { MLFLOW_TRACKING_URI = "http://localhost:5000" }
```

#### Start Services
```bash
# Terminal 1 - MLflow tracking server + UI
uv run mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root data/mlflow/artifacts \
  --serve-artifacts

# Terminal 2 - Marimo with MCP
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run marimo edit examples/ --mcp --no-token --port 2718

# Terminal 3 - MLflow MCP (optional, for IDEs that need stdio)
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run mlflow mcp run
```

VSCode users: each command above is also a task in
[`.vscode/tasks.json`](../.vscode/tasks.json) (Ctrl+Shift+P →
"Tasks: Run Task").

#### Stop Services

`Ctrl+C` in each terminal. To kill stragglers by port:

```bash
# Linux/macOS
lsof -ti :5000 :2718 | xargs kill

# Windows (PowerShell)
Get-NetTCPConnection -LocalPort 5000,2718 |
  Select-Object -ExpandProperty OwningProcess |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

---

### Claude Desktop

Claude Desktop has **full MCP support** out of the box.

#### 1. Locate the config

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

#### 2. Add the marimo-flow MCP servers

```json
{
  "mcpServers": {
    "marimo": {
      "transport": "http",
      "url": "http://localhost:2718/mcp/server"
    },
    "context7": {
      "transport": "sse",
      "url": "https://context7.com/api/v1/mcp/sse"
    },
    "mlflow": {
      "transport": "stdio",
      "command": "mlflow",
      "args": ["mcp", "run"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
      }
    }
  }
}
```

The same config (with one extra `serena` entry) lives at
[`.vscode/mcp.json`](../.vscode/mcp.json) for Claude Code / VS Code
users — copy that as a starting point.

#### 3. Test the connection

1. Make sure both terminals from
   [Quick Start](#1-start-development-environment) are still running
2. Restart Claude Desktop
3. In Claude Desktop, try:
   ```
   List all active marimo notebooks
   Search MLflow experiments
   Get documentation for polars DataFrame
   ```

---

### VSCode

VSCode has **built-in Marimo extension** support.

#### Configuration

**`.vscode/settings.json`** (already configured):
```json
{
  "marimo.autoStartServer": true,
  "marimo.pythonPath": "${workspaceFolder}/.venv/bin/python",
  "marimo.port": 2718,
  "marimo.serverArgs": ["--mcp"],

  "terminal.integrated.env.linux": {
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
    "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src:${workspaceFolder}/snippets"
  }
}
```

#### VSCode Tasks

**`.vscode/tasks.json`** provides shortcuts:

| Task | Shortcut | Description |
|------|----------|-------------|
| Start All Services | `Ctrl+Shift+B` | Start MLflow + Marimo |
| Start MLflow Server | Task menu | MLflow only |
| Start Marimo with MCP | Task menu | Marimo only |
| Run Tests | `Ctrl+Shift+T` | Run pytest |

#### Usage

1. **Open VSCode** in project root
2. **Run Task**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Start All Services"
3. **Open Notebook**: Click `.py` file in `examples/`
4. **Marimo Auto-starts** in embedded browser

#### Marimo Extension Features
- Embedded browser for notebooks
- Auto-start server on file open
- MCP enabled by default (`--mcp` flag)
- Python environment auto-detection

---

### Cursor

Cursor has **advanced AI features** with custom rules support.

#### Configuration Files

**`.cursor/settings.json`** (already configured):
```json
{
  "cursor.chat.model": "claude-sonnet-4.5",
  "cursor.rules": [
    "Prefer Polars over Pandas",
    "Use Altair for visualizations",
    "Marimo cells must be idempotent",
    "Check MLflow before creating experiments"
  ]
}
```

**`.cursorrules`** (already configured):
- Project-specific AI instructions
- MCP tools usage guidelines
- Code standards and best practices
- Common pitfalls to avoid

#### Usage

1. **Open Cursor** in project root
2. **Start Services**: see [Quick Start](#1-start-development-environment)
3. **Use Cursor AI** (`Cmd+K` or `Cmd+L`):
   ```
   @marimo List active notebooks
   @docs How do I use polars lazy evaluation?
   @mlflow Show me the best experiment run
   ```

#### Cursor AI Features
- **Chat Model**: Claude Sonnet 4.5
- **Custom Rules**: Project-specific guidelines in `.cursorrules`
- **MCP Context**: Knows about marimo-flow structure
- **Smart Suggestions**: Polars > Pandas, Altair charts, MLflow tracking

---

## MCP Tools Reference

### Marimo MCP Tools

#### `get_active_notebooks`
Returns list of currently open notebooks.

**Parameters**: None

**Returns**:
```json
{
  "notebooks": [
    "01_pina_poisson_solver.py",
    "lab.py"
  ]
}
```

#### `get_notebook_errors`
Find runtime errors in notebooks.

**Parameters**:
- `notebook_name` (optional): Specific notebook to check

**Returns**:
```json
{
  "errors": [
    {
      "cell_id": "c123",
      "error_type": "ValueError",
      "message": "Invalid data shape",
      "traceback": "..."
    }
  ]
}
```

#### `get_cell_runtime_data`
Inspect cell execution state.

**Parameters**:
- `notebook_name`: Notebook to inspect
- `cell_id`: Cell identifier

**Returns**:
```json
{
  "status": "completed",
  "runtime_ms": 250,
  "outputs": ["DataFrame with 1000 rows"]
}
```

#### `get_tables_and_variables`
Analyze DataFrames and variables.

**Parameters**:
- `notebook_name`: Notebook to inspect

**Returns**:
```json
{
  "tables": [
    {
      "name": "data_raw",
      "type": "polars.DataFrame",
      "shape": [1000, 5],
      "columns": ["id", "age", "income", "category", "score"]
    }
  ]
}
```

### Context7 MCP Tools

#### `search_docs`
Search documentation for Python libraries.

**Parameters**:
- `query`: Search query (e.g., "polars window functions")

**Returns**:
```json
{
  "results": [
    {
      "title": "Window Functions in Polars",
      "url": "...",
      "snippet": "Use .rolling_mean() for moving averages..."
    }
  ]
}
```

#### `get_library_docs`
Get comprehensive library reference.

**Parameters**:
- `library`: Library name (e.g., "polars", "plotly")
- `topic` (optional): Specific topic

**Returns**: Full documentation with code examples

### MLflow MCP Tools

#### `search_experiments`
Find ML experiments.

**Parameters**:
- `filter_string` (optional): MLflow filter expression

**Returns**:
```json
{
  "experiments": [
    {
      "experiment_id": "1",
      "name": "pina-walrus-solver",
      "artifact_location": "..."
    }
  ]
}
```

#### `search_runs`
Query training runs.

**Parameters**:
- `experiment_ids`: List of experiment IDs
- `filter_string` (optional): Filter expression
- `order_by` (optional): Sort order

**Returns**:
```json
{
  "runs": [
    {
      "run_id": "abc123",
      "metrics": {"accuracy": 0.95},
      "params": {"learning_rate": 0.001}
    }
  ]
}
```

#### `log_metric`
Track experiment metric.

**Parameters**:
- `run_id`: Run identifier
- `key`: Metric name
- `value`: Metric value
- `timestamp` (optional): Unix timestamp

**Returns**: Success confirmation

---

## Troubleshooting

### Marimo MCP Server Not Responding

**Symptom**: `curl http://localhost:2718/mcp/server` fails

**Solutions**:
```bash
# 1. Check if marimo is running
ps aux | grep marimo

# 2. Check if marimo has --mcp flag
# Should see: marimo edit examples/ --mcp

# 3. Restart marimo with MCP
pkill -f marimo
marimo edit examples/ --mcp --port 2718

# 4. Check logs
tail -f marimo.log
```

### MLflow Server Not Starting

**Symptom**: `curl http://localhost:5000/health` fails

**Solutions**:
```bash
# 1. Check if port is in use
lsof -i :5000

# 2. Check database path
ls -la data/mlflow/db/mlflow.db

# 3. Create database if missing
mkdir -p data/mlflow/db
touch data/mlflow/db/mlflow.db

# 4. Restart MLflow (Ctrl+C in its terminal, then re-run)
uv run mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root data/mlflow/artifacts --serve-artifacts
```

### MLflow MCP Server Fails

**Symptom**: `mlflow mcp run` exits immediately

**Solutions**:
```bash
# 1. Check MLFLOW_TRACKING_URI is set
echo $MLFLOW_TRACKING_URI
# Should be: http://localhost:5000

# 2. Export it if missing
export MLFLOW_TRACKING_URI=http://localhost:5000

# 3. Verify MLflow is running
curl http://localhost:5000/health

# 4. Try again
mlflow mcp run
```

### Claude Desktop Not Detecting MCP

**Symptom**: MCP tools not available in Claude Desktop

**Solutions**:
```bash
# 1. Verify config file location
# macOS:
ls -la ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux:
ls -la ~/.config/Claude/claude_desktop_config.json

# 2. Validate JSON syntax
cat ~/.config/Claude/claude_desktop_config.json | jq .

# 3. Check services are running (both MLflow on :5000 and Marimo on :2718)
curl -s http://localhost:5000/health && curl -sI http://localhost:2718

# 4. Restart Claude Desktop completely
# Quit app, then reopen

# 5. Check Claude Desktop logs
# macOS:
tail -f ~/Library/Logs/Claude/main.log
```

---

## Advanced Usage

### Custom MCP Server

Add your own MCP server to `.marimo.toml`:

```toml
[mcp.mcpServers.custom-server]
command = "npx"
args = ["-y", "@your-org/your-mcp-server"]
env = { API_KEY = "your-key" }
```

### MCP with Docker

MCP servers work in Docker too (see `docker/.marimo.toml`):

```bash
# Start with docker-compose
docker compose -f docker/docker-compose.yaml up

# MCP endpoints:
# Marimo: http://localhost:2718/mcp/server
# MLflow: Configure with MLFLOW_TRACKING_URI=http://localhost:5000
```

### MCP Debugging

Enable verbose logging:

```bash
# Marimo MCP
MARIMO_DEBUG=1 marimo edit examples/ --mcp

# MLflow MCP
MLFLOW_MCP_DEBUG=1 mlflow mcp run
```

---

## Resources

- [Marimo MCP Documentation](https://docs.marimo.io/guides/editor_features/mcp/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MLflow MCP Guide](https://mlflow.org/docs/latest/mcp.html)
- [Context7 Documentation](https://context7.com/docs)

---

## Summary

**Quick Commands**:
```bash
# Terminal A — MLflow
uv run mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root data/mlflow/artifacts --serve-artifacts

# Terminal B — Marimo with MCP
uv run marimo edit examples/ --mcp --no-token --port 2718

# Stop: Ctrl+C in each terminal
```

For Claude Desktop, copy [`.vscode/mcp.json`](../.vscode/mcp.json) into
`claude_desktop_config.json` (paths above) and restart the app.

**MCP Endpoints**:
- Marimo: `http://localhost:2718/mcp/server`
- Context7: `https://context7.com/api/v1/mcp/sse`
- MLflow: stdio (`mlflow mcp run`)

**Next Steps**:
1. Start services
2. Configure your IDE
3. Try MCP tools with Claude
4. Read [MCP Tools Reference](#mcp-tools-reference)
