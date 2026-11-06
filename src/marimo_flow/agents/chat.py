"""marimo chat adapter — bridges mo.ui.chat to the lead agent's stream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def lead_chat(*, model=None, deps: FlowDeps | None = None) -> Callable:
    """Return an async-generator function suitable for `mo.ui.chat(...)`.

    The returned callable has the signature `(messages, config) -> AsyncIterator[str]`
    expected by marimo's chat component. Uses `run.stream_text(delta=True)` so
    only plain string deltas cross the boundary (avoids JSON-serialisation
    circular-reference errors when MLflow autolog wraps pydantic-ai events).
    """
    run_deps = deps or FlowDeps()
    agent = build_lead_agent(model=model, deps=run_deps)

    async def _chat(messages, config) -> AsyncIterator[str]:  # noqa: ARG001
        user_text = messages[-1].content
        async with agent.run_stream(user_text, deps=run_deps) as run:
            async for delta in run.stream_text(delta=True):
                yield delta

    return _chat
