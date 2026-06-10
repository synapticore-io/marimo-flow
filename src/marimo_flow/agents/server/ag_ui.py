"""AG-UI server — `marimo-flow` lead agent exposed over the AG-UI protocol.

Built as a bare Starlette app whose single POST route dispatches to
`AGUIAdapter.dispatch_request` — the replacement for the deprecated
`Agent.to_ag_ui()` / `AGUIApp` (both removed in pydantic-ai 2.0).
"""

from __future__ import annotations

from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def build_ag_ui_app(*, model=None, deps: FlowDeps | None = None, debug: bool = False):
    agent = build_lead_agent(model=model, deps=deps)
    run_deps = deps or FlowDeps()

    async def run_agent(request: Request) -> Response:
        return await AGUIAdapter.dispatch_request(request, agent=agent, deps=run_deps)

    return Starlette(debug=debug, routes=[Route("/", run_agent, methods=["POST"])])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_ag_ui_app(), host="0.0.0.0", port=8001)
