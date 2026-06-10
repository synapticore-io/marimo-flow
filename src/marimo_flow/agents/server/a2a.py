"""A2A server — `marimo-flow` lead agent exposed over the Agent2Agent protocol.

`node_skills()` builds the AgentCard `skills` list so external A2A agents
can discover what this team can do (one skill per sub-node role).
"""

from __future__ import annotations

from fasta2a import Skill
from fasta2a.pydantic_ai import agent_to_a2a

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def node_skills() -> list[Skill]:
    """A2A capability skills derived from our sub-nodes — one per role."""
    return [
        Skill(
            id="define_problem",
            name="Define a PINA Problem",
            description="Compose a PDE problem spec (equation, domain, BCs, conditions) tailored to the user request.",
            tags=["pina", "pde", "problem"],
            examples=["Define a 1D Poisson on [0,1] with u(0)=u(1)=0."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="define_architecture",
            name="Design a Neural Architecture",
            description="Design a neural-network architecture spec (FNN/FNO/KAN/DeepONet or custom) tailored to a registered Problem.",
            tags=["pina", "architecture", "model"],
            examples=["Pick an FNO with 16 modes and width 32 for 1D Burgers."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="define_solver",
            name="Configure a PINA Solver + Trainer",
            description="Define a Solver (PINN/SAPINN/GAROM/custom) and Trainer config tailored to the registered Problem and Model.",
            tags=["pina", "solver", "training"],
            examples=["Set up a PINN with Adam(lr=1e-3) for 5000 epochs."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="query_mlflow",
            name="Query MLflow",
            description="Inspect, log, and register MLflow runs and models for this team's experiments.",
            tags=["mlflow", "tracking", "registry"],
            examples=["Show the latest run's loss curve."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="edit_notebook",
            name="Edit the marimo Notebook",
            description="Inspect, create, modify, or run cells in the active marimo notebook via the marimo MCP server.",
            tags=["marimo", "notebook", "mcp"],
            examples=["Add a cell that plots the solver loss curve."],
            input_modes=["text"],
            output_modes=["text"],
        ),
    ]


def build_a2a_app(
    *,
    model=None,
    deps: FlowDeps | None = None,
    name: str = "marimo-flow-pina-team",
    description: str = "PINA Physics-Informed NN team (route + notebook + problem + model + solver + mlflow)",
    version: str = "0.1.0",
    url: str = "http://localhost:8000",
    debug: bool = False,
):
    agent = build_lead_agent(model=model, deps=deps)
    return agent_to_a2a(
        agent,
        name=name,
        description=description,
        version=version,
        url=url,
        debug=debug,
        skills=node_skills(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_a2a_app(), host="0.0.0.0", port=8000)
