"""Drive the PINA team graph and snapshot FlowState to MLflow.

The builder-based pydantic-graph runner has no persistence hook (the legacy
`BaseStatePersistence` machinery was deprecated alongside the `Graph` runner),
so state snapshots — resuming a chat == resuming an MLflow run — are logged
explicitly here: one JSON artifact under ``agent_state/`` per node transition.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import mlflow

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.state import FlowState

ARTIFACT_DIR = "agent_state"

# Characters not allowed in Windows filenames (NTFS reserves <>:"/\|?*).
# Builder node IDs are plain names (no colon), but snapshot labels are kept
# defensively sanitized so the same artifact path works on Windows and POSIX.
_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(label: str) -> str:
    """Replace filesystem-reserved chars so the path works cross-platform."""
    return _WIN_INVALID.sub("_", label)


def _log_state(
    client: mlflow.MlflowClient, run_id: str, label: str, state: FlowState
) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"{_safe_filename(label)}.json"
        path.write_text(json.dumps(state.to_jsonable(), default=str, indent=2))
        client.log_artifact(run_id, str(path), artifact_path=ARTIFACT_DIR)


async def run_graph(
    state: FlowState,
    deps: FlowDeps,
    *,
    snapshot_run_id: str | None = None,
) -> str:
    """Run the team graph to completion and return the final summary string.

    When ``snapshot_run_id`` is given, a FlowState JSON artifact is logged
    under ``agent_state/`` after each node transition and once at the end.
    """
    graph = build_graph()
    if snapshot_run_id is None:
        return str(await graph.run(inputs=start_node(), state=state, deps=deps))

    client = mlflow.MlflowClient()
    async with graph.iter(inputs=start_node(), state=state, deps=deps) as run:
        async for _ in run:
            _log_state(client, snapshot_run_id, f"node-{state.last_node}", state)
        output = run.output
    _log_state(client, snapshot_run_id, "end", state)
    return str(output)
