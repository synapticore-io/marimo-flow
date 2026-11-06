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
    from marimo_flow.agents.deps import (
        DEFAULT_MODELS,
        resolve_marimo_mcp_url,
        resolve_mlflow_tracking_uri,
    )

    mlflow_uri = mo.ui.text(resolve_mlflow_tracking_uri(), label="MLflow tracking URI")
    marimo_mcp = mo.ui.text(resolve_marimo_mcp_url(), label="marimo MCP URL")
    mo.vstack([mlflow_uri, marimo_mcp])
    return DEFAULT_MODELS, marimo_mcp, mlflow_uri


@app.cell
def _(DEFAULT_MODELS, marimo_mcp, mlflow_uri, mo):
    from marimo_flow.agents import FlowDeps, build_graph, lead_chat

    # Models come from config.yaml / MARIMO_FLOW_MODEL_<ROLE> / DEFAULT_MODELS.
    deps = FlowDeps(
        mlflow_tracking_uri=mlflow_uri.value,
        marimo_mcp_url=marimo_mcp.value,
    )
    unresolved = {}
    for _role in deps.models:
        try:
            deps.model_for(_role)
        except Exception as _exc:  # missing API key / provider SDK / unknown provider
            unresolved[_role] = _exc
    mo.stop(
        bool(unresolved),
        mo.callout(
            mo.md(
                "**Some roles have no usable model.** Set `models.<role>` in "
                "`config.yaml` or `MARIMO_FLOW_MODEL_<ROLE>`, plus the "
                "provider's API key env var.\n\n"
                + "\n".join(
                    f"- `{r}`: `{deps.models[r]}` (default `{DEFAULT_MODELS[r]}`) → {e}"
                    for r, e in unresolved.items()
                )
            ),
            kind="danger",
        ),
    )
    chat_fn = lead_chat(deps=deps)
    graph = build_graph()
    mo.ui.table(
        [{"role": r, "model": s} for r, s in deps.models.items()],
        label="Resolved models",
        selection=None,
    )
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
