"""Graph assembly — TriageNode is the start, RouteNode dispatches the rest.

Built on the builder-based `GraphBuilder` API. The v1 `BaseNode` subclasses
in `nodes/` are registered as-is via `g.node(...)` — `BaseNode`/`End`/
`GraphRunContext` survive into pydantic-graph v2, only the legacy `Graph`
runner + its persistence machinery were deprecated.
"""

from __future__ import annotations

from pydantic_graph import GraphBuilder
from pydantic_graph.graph_builder import Graph

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes import mlflow_node as _mlflow_node_mod
from marimo_flow.agents.nodes import model as _model_mod
from marimo_flow.agents.nodes import notebook as _notebook_mod
from marimo_flow.agents.nodes import problem as _problem_mod
from marimo_flow.agents.nodes import route as _route_mod
from marimo_flow.agents.nodes import solver as _solver_mod
from marimo_flow.agents.nodes import training as _training_mod
from marimo_flow.agents.nodes import triage as _triage_mod
from marimo_flow.agents.nodes import validation as _validation_mod
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.nodes.route import RouteNode
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.nodes.training import TrainingNode
from marimo_flow.agents.nodes.triage import TriageNode
from marimo_flow.agents.nodes.validation import ValidationNode
from marimo_flow.agents.state import FlowState


def build_graph() -> Graph[FlowState, FlowDeps, TriageNode, str]:
    # Inject the `if TYPE_CHECKING` forward refs into each node module's
    # globals so `GraphBuilder.node()` -> `get_type_hints(cls.run)` resolves
    # the `-> RouteNode` / specialist-union return hints at runtime. The
    # source-level imports stay behind `TYPE_CHECKING` to keep the module
    # graph cycle-free.
    for mod in (
        _triage_mod,
        _notebook_mod,
        _problem_mod,
        _model_mod,
        _solver_mod,
        _training_mod,
        _validation_mod,
        _mlflow_node_mod,
    ):
        mod.__dict__.setdefault("RouteNode", RouteNode)
    _route_mod.__dict__.update(
        NotebookNode=NotebookNode,
        ProblemNode=ProblemNode,
        ModelNode=ModelNode,
        SolverNode=SolverNode,
        TrainingNode=TrainingNode,
        ValidationNode=ValidationNode,
        MLflowNode=MLflowNode,
    )

    g = GraphBuilder[FlowState, FlowDeps, TriageNode, str](
        name="pina-team",
        state_type=FlowState,
        deps_type=FlowDeps,
        input_type=TriageNode,
        output_type=str,
        auto_instrument=False,
    )
    g.add(
        g.edge_from(g.start_node).to(TriageNode),
        g.node(TriageNode),
        g.node(RouteNode),
        g.node(NotebookNode),
        g.node(ProblemNode),
        g.node(ModelNode),
        g.node(SolverNode),
        g.node(TrainingNode),
        g.node(ValidationNode),
        g.node(MLflowNode),
    )
    return g.build()


def start_node() -> TriageNode:
    return TriageNode()
