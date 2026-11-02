"""Plugin-local registry for the active Linear Agent adapter.

A standalone plugin is loaded under one package identity per installation
mode, so it can keep runtime state locally instead of mutating Hermes' global
tool registry.

Two adapters can exist in one process:

* the gateway adapter, constructed and ``connect()``ed by the gateway — it owns
  the webhook listener and registers itself here via ``set_active_adapter``;
* a lazy fallback, built on first use by any process that never ran
  ``connect()`` (desktop chat, CLI, cron). The tools only read ``_client`` and
  ``_mutation_policy``, so the fallback is constructed WITHOUT a webhook
  server, platform lock, or viewer discovery. It reuses the gateway's credential
  chain (YAML, env/secret scope, cached ``plugin-state/linear-agent/oauth.json``)
  and only mints a token when the cached one is expired.

A connected gateway adapter always wins: ``get_active_adapter`` prefers it, and
``set_active_adapter`` replaces the fallback as the resolved adapter.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.config import PlatformConfig

    from .adapter import LinearAgentAdapter

logger = logging.getLogger(__name__)

_active_adapter: LinearAgentAdapter | None = None
_fallback_adapter: LinearAgentAdapter | None = None
_fallback_lock = threading.Lock()

_MISSING_CREDENTIALS = (
    "linear_agent is not connected and this process has no Linear credentials. "
    "Set LINEAR_AGENT_ACCESS_TOKEN, or LINEAR_AGENT_CLIENT_ID and "
    "LINEAR_AGENT_CLIENT_SECRET, for this profile (or run the gateway so the "
    "platform adapter connects)."
)


def set_active_adapter(adapter: LinearAgentAdapter | None) -> None:
    """Register or clear the currently connected Linear Agent adapter."""
    global _active_adapter
    _active_adapter = adapter


def get_active_adapter() -> LinearAgentAdapter:
    """Return the connected adapter, or a lazily built fallback adapter.

    Raises ``RuntimeError`` naming the required env vars when this process has
    neither an adapter nor a usable Linear credential.
    """
    if _active_adapter is not None:
        return _active_adapter
    return _get_fallback_adapter()


def _get_fallback_adapter() -> LinearAgentAdapter:
    global _fallback_adapter
    if _fallback_adapter is None:
        with _fallback_lock:
            if _fallback_adapter is None:
                _fallback_adapter = _build_fallback_adapter()
    return _fallback_adapter


def _reset_fallback_adapter() -> None:
    """Drop the cached fallback (tests, and after a credential change)."""
    global _fallback_adapter
    with _fallback_lock:
        _fallback_adapter = None


def _linear_platform_config() -> PlatformConfig:
    """The effective ``PlatformConfig`` the gateway would build for linear_agent.

    ``load_gateway_config()`` is the documented loader cron and other
    non-gateway processes already use: it merges config.yaml, the plugin's
    ``_apply_yaml_config``/``_env_enablement`` hooks, and env/.env overrides.
    The platform key is matched by value so this works whether or not the
    dynamic ``Platform`` enum member is registered yet.
    """
    from gateway.config import PlatformConfig, load_gateway_config

    try:
        config = load_gateway_config()
        for platform, platform_config in (config.platforms or {}).items():
            if str(getattr(platform, "value", platform)) == "linear_agent":
                return platform_config
    except Exception:  # noqa: BLE001 - a broken config.yaml must not hide env credentials
        logger.debug("[linear_agent] Could not load gateway config for the fallback adapter", exc_info=True)
    # No YAML block (or the loader failed): the adapter constructor still reads
    # LINEAR_AGENT_* from the profile's env / secret scope.
    return PlatformConfig(enabled=True)


def _build_fallback_adapter() -> LinearAgentAdapter:
    from .adapter import LinearAgentAdapter, LinearAgentConfigError, validate_config

    config = _linear_platform_config()
    if not validate_config(config):
        raise RuntimeError(_MISSING_CREDENTIALS)
    try:
        # Deliberately no connect(): no webhook listener, no platform lock, no
        # viewer query. Tools only read ``_client`` and ``_mutation_policy``.
        adapter = LinearAgentAdapter(config)
    except LinearAgentConfigError:
        raise
    except ValueError as exc:
        # ``Platform("linear_agent")`` is a dynamic enum member created by the
        # plugin's ``register()`` pass; without it only the module is loaded.
        raise RuntimeError(
            "linear_agent is loaded but not registered as a Hermes platform, so the "
            "adapter cannot be constructed. Enable the Linear Agent plugin for this "
            "profile (hermes plugins enable linear-agent)."
        ) from exc
    logger.debug("[linear_agent] Built a fallback adapter for a non-gateway process")
    return adapter
