"""Tests for FlowDeps and the provider-agnostic model factory."""

import os

import pytest

from marimo_flow.agents.deps import (
    DEFAULT_MARIMO_MCP_URL,
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_MODELS,
    FlowDeps,
    get_model,
    load_config,
    resolve_marimo_mcp_url,
    resolve_mlflow_tracking_uri,
    resolve_models,
)


def test_default_models_cover_all_roles():
    expected_roles = {
        "route",
        "triage",
        "notebook",
        "problem",
        "model",
        "solver",
        "training",
        "validation",
        "mlflow",
        "lead",
    }
    assert set(DEFAULT_MODELS.keys()) == expected_roles


def test_default_role_models_use_provider_prefix():
    # Shipped defaults all target Ollama Cloud.
    for role, spec in DEFAULT_MODELS.items():
        assert spec.startswith("ollama:"), f"{role} missing provider prefix: {spec!r}"
    assert DEFAULT_MODELS["route"] == "ollama:gemma4:31b-cloud"
    assert DEFAULT_MODELS["lead"] == "ollama:kimi-k2.5:cloud"


def test_get_model_returns_pydantic_ai_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = get_model("route")
    # pydantic-ai's OllamaModel wraps the Ollama-Cloud model name verbatim,
    # including the colon in "gemma4:31b-cloud".
    assert model.model_name == "gemma4:31b-cloud"


def test_get_model_respects_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = get_model("solver", override="ollama:custom:tag")
    assert model.model_name == "custom:tag"


def test_flow_deps_registry_starts_empty():
    deps = FlowDeps()
    assert deps.registry == {}
    assert set(deps.models.keys()) == set(DEFAULT_MODELS.keys())


def test_flow_deps_registry_round_trip():
    deps = FlowDeps()
    sentinel = object()
    deps.registry["mlflow:/some/uri"] = sentinel
    assert deps.registry["mlflow:/some/uri"] is sentinel


def test_env_var_overrides_role(monkeypatch):
    monkeypatch.setenv("MARIMO_FLOW_MODEL_LEAD", "anthropic:claude-sonnet-4-6")
    models = resolve_models(config={})
    assert models["lead"] == "anthropic:claude-sonnet-4-6"
    # Unaffected roles keep the default.
    assert models["route"] == DEFAULT_MODELS["route"]


def test_config_overrides_role(monkeypatch):
    monkeypatch.delenv("MARIMO_FLOW_MODEL_LEAD", raising=False)
    cfg = {"models": {"lead": "openai:gpt-5"}}
    models = resolve_models(config=cfg)
    assert models["lead"] == "openai:gpt-5"


def test_env_var_wins_over_config(monkeypatch):
    monkeypatch.setenv("MARIMO_FLOW_MODEL_LEAD", "groq:llama-3.3-70b-versatile")
    cfg = {"models": {"lead": "openai:gpt-5"}}
    assert resolve_models(config=cfg)["lead"] == "groq:llama-3.3-70b-versatile"


def test_env_block_seeds_missing_env(monkeypatch):
    monkeypatch.delenv("SOME_FLAG_FOR_TEST", raising=False)
    cfg = {"env": {"SOME_FLAG_FOR_TEST": "abc"}, "models": {}}
    resolve_models(config=cfg)
    assert os.environ["SOME_FLAG_FOR_TEST"] == "abc"


def test_env_block_does_not_overwrite_existing(monkeypatch):
    monkeypatch.setenv("SOME_FLAG_FOR_TEST", "already-set")
    cfg = {"env": {"SOME_FLAG_FOR_TEST": "new"}, "models": {}}
    resolve_models(config=cfg)
    assert os.environ["SOME_FLAG_FOR_TEST"] == "already-set"


def test_load_config_returns_empty_without_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MARIMO_FLOW_CONFIG", raising=False)
    assert load_config() == {}


def test_load_config_reads_yaml(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("models:\n  lead: openai:gpt-5\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MARIMO_FLOW_CONFIG", raising=False)
    data = load_config()
    assert data == {"models": {"lead": "openai:gpt-5"}}


def test_load_config_honours_marimo_flow_config_env(monkeypatch, tmp_path):
    cfg_path = tmp_path / "custom-spot.yaml"
    cfg_path.write_text(
        "models:\n  solver: anthropic:claude-sonnet-4-6\n", encoding="utf-8"
    )
    monkeypatch.setenv("MARIMO_FLOW_CONFIG", str(cfg_path))
    monkeypatch.chdir(tmp_path)
    assert load_config()["models"]["solver"] == "anthropic:claude-sonnet-4-6"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    with pytest.raises(ValueError):
        get_model("route", override="no-such-provider:foo")


def test_mlflow_tracking_uri_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_mlflow_tracking_uri(config={}) == DEFAULT_MLFLOW_TRACKING_URI


def test_mlflow_tracking_uri_from_config(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = {"mlflow": {"tracking_uri": "http://mlflow.internal:5000"}}
    assert resolve_mlflow_tracking_uri(config=cfg) == "http://mlflow.internal:5000"


def test_mlflow_tracking_uri_env_wins_over_config(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env-wins.db")
    cfg = {"mlflow": {"tracking_uri": "http://config.local:5000"}}
    assert resolve_mlflow_tracking_uri(config=cfg) == "sqlite:///env-wins.db"


def test_marimo_mcp_url_default(monkeypatch):
    monkeypatch.delenv("MARIMO_MCP_URL", raising=False)
    assert resolve_marimo_mcp_url(config={}) == DEFAULT_MARIMO_MCP_URL


def test_marimo_mcp_url_from_config(monkeypatch):
    monkeypatch.delenv("MARIMO_MCP_URL", raising=False)
    cfg = {"marimo": {"mcp_url": "http://marimo.internal:2718/mcp/server"}}
    assert (
        resolve_marimo_mcp_url(config=cfg) == "http://marimo.internal:2718/mcp/server"
    )


def test_marimo_mcp_url_env_wins_over_config(monkeypatch):
    monkeypatch.setenv("MARIMO_MCP_URL", "http://env-wins.local/mcp")
    cfg = {"marimo": {"mcp_url": "http://config.local/mcp"}}
    assert resolve_marimo_mcp_url(config=cfg) == "http://env-wins.local/mcp"


def test_flow_deps_resolves_uris(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///flowdeps-test.db")
    monkeypatch.setenv("MARIMO_MCP_URL", "http://flowdeps-test:2718/mcp/server")
    deps = FlowDeps()
    assert deps.mlflow_tracking_uri == "sqlite:///flowdeps-test.db"
    assert deps.marimo_mcp_url == "http://flowdeps-test:2718/mcp/server"


def test_model_for_caches_same_spec(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    deps = FlowDeps()
    a = deps.model_for("route")
    b = deps.model_for("route")
    assert a is b, "FlowDeps.model_for should return the same instance per spec"
    assert deps.model_cache


async def test_aclose_closes_cached_clients(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    deps = FlowDeps()
    # Inject a stub model whose client records close() calls.
    closed = []

    class _StubClient:
        async def close(self):
            closed.append("closed")

    class _StubModel:
        client = _StubClient()

    deps.model_cache["stub:spec"] = _StubModel()  # type: ignore[assignment]
    await deps.aclose()
    assert closed == ["closed"]
    assert deps.model_cache == {}
