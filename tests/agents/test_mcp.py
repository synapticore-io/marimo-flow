"""Tests for MCP toolset builders."""

import pytest

from marimo_flow.agents.mcp import (
    build_marimo_mcp,
    build_mcp_servers,
    build_mlflow_mcp,
)


def test_build_marimo_mcp_uses_default_url():
    server = build_marimo_mcp()
    assert "127.0.0.1:2718" in server.client.transport.url


def test_build_marimo_mcp_respects_url_override():
    server = build_marimo_mcp(url="http://example:9999/mcp/server")
    assert "example:9999" in server.client.transport.url


def test_build_mlflow_mcp_uses_stdio_with_tracking_uri(tmp_path):
    db = tmp_path / "mlruns.db"
    server = build_mlflow_mcp(tracking_uri=f"sqlite:///{db}")
    transport = server.client.transport
    assert transport.command == "mlflow"
    assert transport.args == ["mcp", "run"]
    assert transport.env["MLFLOW_TRACKING_URI"] == f"sqlite:///{db}"


def test_build_mcp_servers_disabled_returns_empty():
    assert build_mcp_servers("disabled") == []


def test_build_mcp_servers_unknown_transport_raises():
    with pytest.raises(ValueError, match="unknown transport"):
        build_mcp_servers("not-a-transport")
