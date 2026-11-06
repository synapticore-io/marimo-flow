"""FlowDeps — in-memory side-channel for live objects + per-role model config.

Model specs are provider-prefixed strings (e.g. ``"ollama:qwen3-coder:480b-cloud"``,
``"anthropic:claude-sonnet-4-6"``, ``"openai:gpt-5"``). Resolution is delegated
to ``pydantic_ai.models.infer_model``, so every pydantic-ai provider works.
The base install ships only the OpenAI SDK; native-SDK providers (anthropic,
google, groq, mistral, ...) need their ``pydantic-ai-slim[<extra>]``.

Configuration resolution (highest wins):

1. Real shell-exported env vars
2. Env vars from a local ``.env`` file (python-dotenv)
3. YAML file — first of these that exists at CWD is used:
   ``$MARIMO_FLOW_CONFIG`` → ``config.yaml`` → ``config.yml`` →
   ``marimo-flow.yaml`` → ``marimo-flow.yml``.
   Supported keys:
     * ``models.<role>: "<provider>:<model>"``
     * ``env.<KEY>: "<value>"``             — seeds missing env vars
     * ``mlflow.tracking_uri: "<uri>"``     — MLflow backend
     * ``marimo.mcp_url: "<url>"``          — marimo MCP endpoint
4. ``DEFAULT_MODELS`` and the hardcoded URI defaults below.

Per-role specs can additionally be overridden by
``MARIMO_FLOW_MODEL_<ROLE>=<spec>`` env vars (the per-role env overrides
always win — layered on top of whatever came from the yaml).

Provider auth uses each provider's standard env var (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``GROQ_API_KEY``, ``OLLAMA_BASE_URL``, ...).
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv
from pydantic_ai.models import Model, infer_model

if TYPE_CHECKING:
    from marimo_flow.agents.state import FlowState

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
# Local-development MLflow defaults — match the docker-compose layout
# (docker/docker-compose.yaml) so the same paths work in container + host.
DEFAULT_MLFLOW_LAYOUT_ROOT = "data/mlflow"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///data/mlflow/db/mlflow.db"
DEFAULT_MLFLOW_ARTIFACT_ROOT = "data/mlflow/artifacts"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "marimo-flow"
DEFAULT_MARIMO_MCP_URL = "http://127.0.0.1:2718/mcp/server"
DEFAULT_PROVENANCE_DB_PATH = "provenance.duckdb"

DEFAULT_MODELS: dict[str, str] = {
    "route": "ollama:gemma4:31b-cloud",
    "triage": "ollama:gemma4:31b-cloud",
    "notebook": "ollama:qwen3-coder:480b-cloud",
    "problem": "ollama:qwen3-coder:480b-cloud",
    "model": "ollama:qwen3.5:cloud",
    "solver": "ollama:qwen3-coder:480b-cloud",
    "training": "ollama:qwen3-coder:480b-cloud",
    "validation": "ollama:qwen3.5:cloud",
    "mlflow": "ollama:gpt-oss:20b-cloud",
    "lead": "ollama:kimi-k2.5:cloud",
}

_CONFIG_CANDIDATES = (
    "config.yaml",
    "config.yml",
    "marimo-flow.yaml",
    "marimo-flow.yml",
)

# Tracks whether dotenv has been attempted this process so repeated config
# lookups don't re-parse the .env every call.
_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load ``.env`` from CWD on first access. Existing env vars always win."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    dotenv_path = os.environ.get("MARIMO_FLOW_DOTENV", ".env")
    load_dotenv(dotenv_path, override=False)
    _DOTENV_LOADED = True


def _config_path() -> Path | None:
    env = os.environ.get("MARIMO_FLOW_CONFIG")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    for name in _CONFIG_CANDIDATES:
        p = Path(name)
        if p.is_file():
            return p
    return None


def load_config() -> dict[str, Any]:
    """Parse the marimo-flow yaml config if present. Empty dict otherwise.

    Also loads ``.env`` (if any) and applies the ``env:`` block from the yaml
    as a side effect, so ``os.environ`` is populated before callers read
    provider auth vars.
    """
    _load_dotenv_once()
    p = _config_path()
    if p is None:
        return {}
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    _apply_env_block(data)
    return data


def _apply_env_block(config: dict[str, Any]) -> None:
    """Seed os.environ from config['env'] without overriding existing values."""
    env_block = config.get("env") or {}
    if not isinstance(env_block, dict):
        return
    for key, value in env_block.items():
        if key not in os.environ and value is not None:
            os.environ[key] = str(value)


def resolve_models(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Role → spec map, overlaying DEFAULT_MODELS with config and env vars."""
    if config is None:
        config = load_config()
    else:
        _apply_env_block(config)
    models = dict(DEFAULT_MODELS)
    yaml_models = config.get("models") or {}
    if isinstance(yaml_models, dict):
        models.update({str(k): str(v) for k, v in yaml_models.items()})
    for role in list(models):
        env_key = f"MARIMO_FLOW_MODEL_{role.upper()}"
        if env_key in os.environ:
            models[role] = os.environ[env_key]
    return models


def resolve_mlflow_tracking_uri(config: dict[str, Any] | None = None) -> str:
    """MLflow URI: env MLFLOW_TRACKING_URI > config.mlflow.tracking_uri > default."""
    if config is None:
        config = load_config()
    env_value = os.environ.get("MLFLOW_TRACKING_URI")
    if env_value:
        return env_value
    mlflow_cfg = config.get("mlflow") or {}
    if isinstance(mlflow_cfg, dict) and mlflow_cfg.get("tracking_uri"):
        return str(mlflow_cfg["tracking_uri"])
    return DEFAULT_MLFLOW_TRACKING_URI


def resolve_marimo_mcp_url(config: dict[str, Any] | None = None) -> str:
    """marimo MCP URL: env MARIMO_MCP_URL > config.marimo.mcp_url > default."""
    if config is None:
        config = load_config()
    env_value = os.environ.get("MARIMO_MCP_URL")
    if env_value:
        return env_value
    marimo_cfg = config.get("marimo") or {}
    if isinstance(marimo_cfg, dict) and marimo_cfg.get("mcp_url"):
        return str(marimo_cfg["mcp_url"])
    return DEFAULT_MARIMO_MCP_URL


def resolve_provenance_db_path(config: dict[str, Any] | None = None) -> str:
    """Provenance DB path: env MARIMO_FLOW_PROVENANCE_DB > config.provenance.db_path > default."""
    if config is None:
        config = load_config()
    env_value = os.environ.get("MARIMO_FLOW_PROVENANCE_DB")
    if env_value:
        return env_value
    prov_cfg = config.get("provenance") or {}
    if isinstance(prov_cfg, dict) and prov_cfg.get("db_path"):
        return str(prov_cfg["db_path"])
    return DEFAULT_PROVENANCE_DB_PATH


def get_model(role: str, *, override: str | None = None) -> Model:
    """Resolve a role to a pydantic-ai ``Model``.

    ``override`` is a spec string that short-circuits the role lookup,
    typically passed as ``override=ctx.deps.models[role]`` from a Node so
    each FlowDeps instance can hold its own per-run model assignment.

    Only ``ollama:`` specs get a local ``OLLAMA_BASE_URL`` default seeded;
    every other provider is resolved untouched by ``infer_model``.
    """
    spec = override or resolve_models()[role]
    if spec.startswith("ollama:"):
        os.environ.setdefault("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    return infer_model(spec)


@dataclass
class FlowDeps:
    """In-memory deps. Not persisted — recreated per session.

    ``registry`` maps MLflow artifact URI → live object (PINA Problem, torch
    model, Trainer). ``FlowState`` only holds the URIs.

    ``state`` is wired in by each Node before calling
    ``agent.run(..., deps=ctx.deps)`` so FunctionToolset tools can reach
    the current FlowState via ``ctx.deps.state``.

    ``models``, ``mlflow_tracking_uri`` and ``marimo_mcp_url`` are populated
    by the config resolver (``.env`` + ``config.yaml`` + env vars) at
    construction time. Override any of them at runtime via direct
    attribute assignment before calling ``graph.run(...)``.

    ``model_cache`` stores resolved pydantic-ai ``Model`` objects keyed by
    spec string so we re-use (and later close) their underlying HTTP
    clients instead of leaking one per Node invocation.
    """

    models: dict[str, str] = field(default_factory=resolve_models)
    registry: dict[str, Any] = field(default_factory=dict)
    mlflow_tracking_uri: str = field(default_factory=resolve_mlflow_tracking_uri)
    marimo_mcp_url: str = field(default_factory=resolve_marimo_mcp_url)
    provenance_db_path: str = field(default_factory=resolve_provenance_db_path)
    state: FlowState | None = None
    model_cache: dict[str, Model] = field(default_factory=dict)
    _provenance_store: Any = None  # lazy ProvenanceStore — never touch disk
    #                                  until someone calls provenance()
    _preset_catalog: Any = None  # lazy PresetCatalog — memory of user compositions

    def provenance(self) -> Any:
        """Return a lazy DuckDB-backed ProvenanceStore.

        Constructed on first access against ``self.provenance_db_path``;
        reused for the lifetime of this FlowDeps. Closed by ``aclose()``.
        Use ``provenance_db_path=":memory:"`` for ephemeral tests.
        """
        if self._provenance_store is None:
            from marimo_flow.agents.services.provenance import ProvenanceStore

            self._provenance_store = ProvenanceStore(self.provenance_db_path)
        return self._provenance_store

    def preset_catalog(self) -> Any:
        """Return a lazy PresetCatalog over the provenance store.

        The catalog is a persistent memory of *user-authored* compositions
        (ProblemSpec / ModelSpec / SolverPlan recipes that worked before).
        No built-in seeding — agents compose from PINA primitives via
        ``services.composer.compose_problem`` and register successful
        compositions here via ``curator_toolset.register_preset``.
        Subsequent sessions can retrieve and clone them.
        """
        if self._preset_catalog is None:
            from marimo_flow.agents.services.preset_catalog import PresetCatalog

            self._preset_catalog = PresetCatalog(self.provenance())
        return self._preset_catalog

    def model_for(self, role: str) -> Model:
        """Return a cached pydantic-ai Model for ``role``.

        Build the Model once per spec and reuse its provider client across
        Node runs — this is what lets ``aclose()`` close every HTTP
        connection on teardown.
        """
        spec = self.models[role]
        cached = self.model_cache.get(spec)
        if cached is not None:
            return cached
        model = get_model(role, override=spec)
        self.model_cache[spec] = model
        return model

    async def aclose(self) -> None:
        """Close every cached model's underlying HTTP client.

        Call this at the end of ``graph.run(...)`` (or from an
        ``AsyncExitStack``) to release sockets cleanly and silence the
        ResourceWarnings we otherwise get at interpreter shutdown.
        """
        seen: set[int] = set()
        for model in self.model_cache.values():
            client = getattr(model, "client", None) or getattr(model, "_client", None)
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            closer = getattr(client, "close", None) or getattr(client, "aclose", None)
            if closer is None:
                continue
            result = closer()
            # Both sync and async .close() variants exist across providers.
            if hasattr(result, "__await__"):
                with contextlib.suppress(Exception):
                    await result
        self.model_cache.clear()
        if self._provenance_store is not None:
            with contextlib.suppress(Exception):
                self._provenance_store.close()
            self._provenance_store = None
