"""MCP toolset builders for the agents.

Two purpose-built helpers point at the project's existing MCP servers
(see .vscode/mcp.json):
  * marimo MCP at http://127.0.0.1:2718/mcp/server (HTTP)
  * mlflow MCP via `mlflow mcp run` (stdio)

`build_mcp_servers()` is a generic transport-selector kept for ad-hoc
use — adapted from `marimo-agent/rag_marimo_agent.py:322-340`.
"""

from __future__ import annotations

from fastmcp.client.transports import (
    SSETransport,
    StdioTransport,
    StreamableHttpTransport,
)
from pydantic_ai.mcp import MCPToolset

DEFAULT_MARIMO_MCP_URL = "http://127.0.0.1:2718/mcp/server"


def build_marimo_mcp(
    url: str = DEFAULT_MARIMO_MCP_URL,
    *,
    startup_timeout: float = 30.0,
) -> MCPToolset:
    """Build the marimo MCP streamable-HTTP toolset.

    ``startup_timeout`` widens the handshake window from FastMCP's default —
    marimo's ``--mcp`` endpoint can take a few seconds to register tools when
    a notebook has many cells.
    """
    return MCPToolset(StreamableHttpTransport(url=url), init_timeout=startup_timeout)


def build_mlflow_mcp(
    tracking_uri: str = "sqlite:///data/mlflow/db/mlflow.db",
    *,
    startup_timeout: float = 60.0,
) -> MCPToolset:
    """Build the MLflow MCP stdio toolset.

    ``startup_timeout`` is the handshake window for ``mlflow mcp run``. MLflow
    initialises the store on startup (SQLite migrations, plugin loading) which
    can exceed FastMCP's default on a cold run.
    """
    transport = StdioTransport(
        command="mlflow",
        args=["mcp", "run"],
        env={"MLFLOW_TRACKING_URI": tracking_uri},
    )
    return MCPToolset(transport, init_timeout=startup_timeout)


def build_mcp_servers(
    transport: str,
    *,
    cmd: str = "deno",
    args: str = "",
    url: str = "",
) -> list[MCPToolset]:
    if transport == "disabled":
        return []
    if transport == "stdio":
        arg_list = [a for a in args.split(" ") if a]
        return [MCPToolset(StdioTransport(command=cmd, args=arg_list))]
    if transport == "sse":
        return [MCPToolset(SSETransport(url=url))]
    if transport == "streamable-http":
        return [MCPToolset(StreamableHttpTransport(url=url))]
    raise ValueError(f"unknown transport: {transport!r}")
