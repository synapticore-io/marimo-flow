"""marimo-flow CLI — entry points for common tasks.

Usage::

    marimo-flow solve "Solve the Burgers equation with a small PINN"
    marimo-flow solve -m anthropic:claude-sonnet-4-6 "..."
    marimo-flow lab                                  # open examples/lab.py in marimo
    marimo-flow config-show                          # print the resolved config
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path

import typer

from marimo_flow.agents.deps import (
    FlowDeps,
    load_config,
    resolve_marimo_mcp_url,
    resolve_mlflow_tracking_uri,
    resolve_models,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="marimo-flow — provider-agnostic PINA agent team.",
)

log = logging.getLogger("marimo_flow.cli")


@app.command("config-show")
def config_show() -> None:
    """Print the resolved config (models, URIs) as JSON."""
    payload = {
        "models": resolve_models(),
        "mlflow_tracking_uri": resolve_mlflow_tracking_uri(),
        "marimo_mcp_url": resolve_marimo_mcp_url(),
        "yaml_config_keys": sorted((load_config() or {}).keys()) or None,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("solve")
def solve(
    intent: str = typer.Argument(..., help="Natural-language task for the team."),
    max_epochs: int = typer.Option(
        100, "--max-epochs", help="Hint for training budget (the LLM may ignore)."
    ),
    n_points: int = typer.Option(
        1000, "--n-points", help="Hint for collocation points per domain."
    ),
    model_override: list[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override a role: --model lead=anthropic:claude-sonnet-4-6 (repeatable).",
    ),
    timeout: float = typer.Option(
        900.0, "--timeout", help="Overall graph timeout in seconds."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full agent graph against the user intent and print the result."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    from marimo_flow.agents.runner import run_graph
    from marimo_flow.agents.state import FlowState

    deps = FlowDeps()
    for kv in model_override or []:
        if "=" not in kv:
            typer.echo(f"--model expects role=spec, got {kv!r}", err=True)
            raise typer.Exit(code=2)
        role, spec = kv.split("=", 1)
        deps.models[role.strip()] = spec.strip()

    typer.echo(f"models: {deps.models}", err=True)
    typer.echo(f"mlflow: {deps.mlflow_tracking_uri}", err=True)

    full_intent = (
        f"{intent}\n(Training hints: max_epochs={max_epochs}, n_points={n_points}.)"
    )
    state = FlowState(user_intent=full_intent)

    async def _run() -> str:
        try:
            return await asyncio.wait_for(run_graph(state, deps), timeout=timeout)
        finally:
            await deps.aclose()

    try:
        output = asyncio.run(_run())
    except TimeoutError:
        typer.echo(
            f"graph timed out after {timeout:.0f}s, last_node={state.last_node}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    typer.echo(f"\n=== summary ===\n{output}")
    typer.echo("\n=== state ===")
    typer.echo(f"  problem_artifact_uri : {state.problem_artifact_uri}")
    typer.echo(f"  model_artifact_uri   : {state.model_artifact_uri}")
    typer.echo(f"  solver_artifact_uri  : {state.solver_artifact_uri}")
    typer.echo(f"  training_artifact_uri: {state.training_artifact_uri}")
    typer.echo(f"  training_run_id      : {state.training_run_id}")
    typer.echo(f"  mlflow_run_id        : {state.mlflow_run_id}")


@app.command("lab")
def lab(
    notebook: Path = typer.Option(
        Path("examples/lab.py"),
        "--notebook",
        "-n",
        help="Marimo notebook to open.",
    ),
    port: int = typer.Option(2718, "--port", "-p"),
    mcp: bool = typer.Option(True, "--mcp/--no-mcp", help="Expose marimo MCP."),
) -> None:
    """Open the lab notebook via the locally installed ``marimo`` binary."""
    if not notebook.exists():
        typer.echo(f"notebook not found: {notebook}", err=True)
        raise typer.Exit(code=2)
    cmd = ["marimo", "edit", str(notebook), "--port", str(port), "--headless"]
    if mcp:
        cmd += ["--mcp", "--no-token"]
    typer.echo("launching: " + " ".join(cmd), err=True)
    raise typer.Exit(code=subprocess.call(cmd))


def main() -> None:  # entry point for ``marimo-flow``
    app()


if __name__ == "__main__":
    main()
