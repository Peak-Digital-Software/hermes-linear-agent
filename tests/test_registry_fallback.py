"""Fallback adapter for processes where the gateway never connected.

Desktop chat, CLI, and cron import the plugin (so the ``linear_agent_*`` tools
exist) but never run ``LinearAgentAdapter.connect()``. The registry must still
hand the tools a configured GraphQL client — without starting a webhook
listener — and a genuinely connected gateway adapter must always win.
"""

from __future__ import annotations

import pytest
from gateway.config import PlatformConfig

from hermes_linear_agent import registry
from hermes_linear_agent.adapter import LinearAgentAdapter


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch, tmp_path):
    """Isolate the process-global registry and the profile config it loads."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry.set_active_adapter(None)
    registry._reset_fallback_adapter()
    yield
    registry.set_active_adapter(None)
    registry._reset_fallback_adapter()


def test_fallback_adapter_is_configured_without_a_web_server(monkeypatch):
    monkeypatch.setenv("LINEAR_AGENT_ACCESS_TOKEN", "lin_test_access_token")

    def _no_server(*_args, **_kwargs):  # pragma: no cover - fires on regression
        raise AssertionError("the fallback adapter must not start a webhook server")

    monkeypatch.setattr("hermes_linear_agent.adapter.web.AppRunner", _no_server)

    adapter = registry.get_active_adapter()

    assert isinstance(adapter, LinearAgentAdapter)
    assert adapter._client.configured
    assert adapter._mutation_policy  # tools read it on every mutation
    assert adapter._runner is None  # never connected
    assert adapter._running is False
    # Built once and reused: a tool call per turn must not rebuild the client.
    assert registry.get_active_adapter() is adapter


def test_connected_gateway_adapter_wins_over_the_fallback(monkeypatch):
    monkeypatch.setenv("LINEAR_AGENT_ACCESS_TOKEN", "lin_test_access_token")
    fallback = registry.get_active_adapter()
    assert isinstance(fallback, LinearAgentAdapter)

    real = LinearAgentAdapter(PlatformConfig(enabled=True, extra={"access_token": "lin_gateway_token"}))
    registry.set_active_adapter(real)
    try:
        assert registry.get_active_adapter() is real
    finally:
        registry.set_active_adapter(None)
    # Clearing the gateway adapter falls back to the cached offline adapter.
    assert registry.get_active_adapter() is fallback


def test_missing_credentials_name_the_required_env_vars(monkeypatch):
    # Strip the credential chain explicitly: ``_str_env`` reads the profile's
    # secret scope, so an operator running the suite locally would otherwise
    # supply a real token through it.
    monkeypatch.setattr(registry, "_linear_platform_config", lambda: PlatformConfig(enabled=True))
    monkeypatch.setattr("hermes_linear_agent.adapter._str_env", lambda _name: "")
    monkeypatch.setattr("hermes_linear_agent.adapter._cached_auth_token", lambda _extra: {})

    with pytest.raises(RuntimeError) as excinfo:
        registry.get_active_adapter()

    message = str(excinfo.value)
    assert "LINEAR_AGENT_ACCESS_TOKEN" in message
    assert "LINEAR_AGENT_CLIENT_ID" in message
    assert "LINEAR_AGENT_CLIENT_SECRET" in message
