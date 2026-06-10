"""Tests for the assembled graph."""

from __future__ import annotations

from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.nodes.route import RouteNode
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.nodes.training import TrainingNode
from marimo_flow.agents.nodes.triage import TriageNode
from marimo_flow.agents.nodes.validation import ValidationNode


def test_graph_contains_all_registered_nodes():
    graph = build_graph()
    expected = {
        cls.__name__
        for cls in (
            TriageNode,
            RouteNode,
            NotebookNode,
            ProblemNode,
            ModelNode,
            SolverNode,
            TrainingNode,
            ValidationNode,
            MLflowNode,
        )
    }
    assert expected.issubset(set(graph.nodes.keys()))


def test_start_node_is_triage():
    assert isinstance(start_node(), TriageNode)


def test_graph_renders_mermaid():
    graph = build_graph()
    code = graph.render()
    assert "TriageNode" in code
    assert "RouteNode" in code
    assert "stateDiagram" in code or "graph" in code.lower()
