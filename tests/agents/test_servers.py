"""Tests for A2A and AG-UI ASGI app builders."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel

from marimo_flow.agents.server.a2a import build_a2a_app, node_skills
from marimo_flow.agents.server.ag_ui import build_ag_ui_app


def test_a2a_app_is_asgi_callable(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    app = build_a2a_app(model=TestModel())
    assert callable(app)


def test_ag_ui_app_is_asgi_callable(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    app = build_ag_ui_app(model=TestModel())
    assert callable(app)


def test_node_skills_cover_all_roles():
    skills = node_skills()
    ids = {s["id"] for s in skills}
    assert ids == {
        "define_problem",
        "define_architecture",
        "define_solver",
        "query_mlflow",
        "edit_notebook",
    }


def test_node_skills_have_required_fields():
    for skill in node_skills():
        assert skill["id"]
        assert skill["name"]
        assert skill["description"]
        assert isinstance(skill["tags"], list) and skill["tags"]
        assert "text" in skill["input_modes"]
        assert "text" in skill["output_modes"]
