"""Tests for the marimo chat adapter."""

from __future__ import annotations

from types import SimpleNamespace

from pydantic_ai.models.test import TestModel

from marimo_flow.agents.chat import lead_chat
from marimo_flow.agents.deps import FlowDeps


async def test_lead_chat_yields_deltas(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    deps = FlowDeps()
    model = TestModel(custom_output_text="hello world", call_tools=[])
    chat_fn = lead_chat(model=model, deps=deps)
    messages = [SimpleNamespace(content="hi")]
    config = None
    chunks = []
    async for delta in chat_fn(messages, config):
        chunks.append(delta)
    assert "".join(chunks).strip() == "hello world"


def test_lead_chat_factory_returns_callable(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    fn = lead_chat(model=TestModel())
    assert callable(fn)


def test_lead_chat_passes_deps_to_lead_agent(monkeypatch):
    seen = {}

    def _fake_build(*, model=None, deps=None):
        seen["deps"] = deps
        return object()

    monkeypatch.setattr("marimo_flow.agents.chat.build_lead_agent", _fake_build)
    deps = FlowDeps(provenance_db_path=":memory:")
    lead_chat(deps=deps)
    assert seen["deps"] is deps
