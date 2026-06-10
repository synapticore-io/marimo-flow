"""marimo-flow PINA team — interactive demo notebook."""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # marimo-flow PINA Team

    Reactive multi-agent demo. The lead agent dispatches to specialists
    (`Notebook`, `Problem`, `Model`, `Solver`, `MLflow`) over a `pydantic-graph`
    state machine. All runs are tracked in MLflow.
    """)
    return


@app.cell
def _(mo):
    base_url = mo.ui.text("http://localhost:11434/v1", label="Ollama base_url")
    mlflow_uri = mo.ui.text("sqlite:///mlruns.db", label="MLflow tracking URI")
    marimo_mcp = mo.ui.text("http://127.0.0.1:2718/mcp/server", label="marimo MCP URL")
    mo.vstack([base_url, mlflow_uri, marimo_mcp])
    return base_url, mlflow_uri, marimo_mcp


@app.cell
def _(base_url, mlflow_uri, marimo_mcp):
    import os

    os.environ["OLLAMA_BASE_URL"] = base_url.value

    from marimo_flow.agents import FlowDeps, build_graph, lead_chat

    deps = FlowDeps(
        mlflow_tracking_uri=mlflow_uri.value,
        marimo_mcp_url=marimo_mcp.value,
    )
    chat_fn = lead_chat(deps=deps)
    graph = build_graph()
    return chat_fn, deps, graph


@app.cell
def _(chat_fn, mo):
    chat = mo.ui.chat(
        chat_fn,
        prompts=[
            "Solve a 1D Poisson equation on [0,1] with u(0)=u(1)=0 using a PINN.",
            "Show me the latest MLflow run for this experiment.",
            "Add a new cell that plots the solver loss curve.",
        ],
        max_height=520,
    )
    chat
    return (chat,)


@app.cell
def _(graph, mo):
    mo.mermaid(graph.render())
    return


if __name__ == "__main__":
    app.run()
