"""Standalone plugin runtime bootstrap for compatibility tests."""

from __future__ import annotations

import pytest
from gateway.platform_registry import PlatformEntry, platform_registry

from hermes_linear_agent import register


class _PlatformRegistrationContext:
    """Forward the plugin platform contract into Hermes' public registry."""

    @staticmethod
    def register_platform(**kwargs):
        platform_registry.register(
            PlatformEntry(
                **kwargs,
                source="plugin",
                plugin_name="linear-agent",
            )
        )

    @staticmethod
    def register_tool(**_kwargs):
        # Tool registration has dedicated contract and dispatch tests. The
        # bootstrap only reproduces the loader ordering required for dynamic
        # Platform("linear_agent") construction on pristine Hermes.
        return None


@pytest.fixture(scope="session", autouse=True)
def _register_standalone_platform():
    register(_PlatformRegistrationContext())


@pytest.fixture(autouse=True)
def _isolate_linear_env(monkeypatch):
    """Tests must not see the operator's real LINEAR_* credentials."""
    import os

    for name in list(os.environ):
        if name.startswith("LINEAR_"):
            monkeypatch.delenv(name, raising=False)
