"""Tests for the lead agent wrapper."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


async def test_lead_agent_exposes_run_pina_workflow_tool(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    from marimo_flow.agents.toolsets.lead import lead_toolset

    deps = FlowDeps()
    agent = build_lead_agent(model=TestModel(), deps=deps)
    # Tool lives on the external lead_toolset attached via toolsets=[...]
    assert "run_pina_workflow" in lead_toolset.tools
    assert lead_toolset in agent.toolsets


async def test_lead_agent_skips_workflow_when_model_responds_directly(monkeypatch):
    """If the LLM produces a text reply (not a tool call), the workflow stays untouched."""
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    deps = FlowDeps()
    agent = build_lead_agent(
        model=TestModel(custom_output_text="hi, no workflow needed", call_tools=[]),
        deps=deps,
    )
    result = await agent.run("just say hi")
    assert "hi" in result.output.lower()


def test_lead_agent_uses_deps_lead_model(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    deps = FlowDeps(provenance_db_path=":memory:")
    deps.models["lead"] = "openai:gpt-5"
    agent = build_lead_agent(deps=deps)
    assert (agent.model.system, agent.model.model_name) == ("openai", "gpt-5")
