#!/usr/bin/env python3
"""Verify that Hermes discovered the standalone Linear Agent plugin."""

from __future__ import annotations

import json

from gateway.platform_registry import platform_registry
from hermes_cli.plugins import get_plugin_manager
from tools.registry import registry

EXPECTED_TOOLS = 59
SMOKE_TOOLS = (
    "linear_agent_update_issue",
    "linear_agent_create_comment",
    "linear_agent_update_plan",
)


def main() -> int:
    manager = get_plugin_manager()
    manager.discover_and_load(force=True)
    matches = [item for item in manager.list_plugins() if item["name"] == "linear-agent"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one linear-agent plugin, found {matches!r}")
    plugin = matches[0]
    if not plugin["enabled"] or plugin["error"]:
        raise RuntimeError(f"plugin failed to load: {plugin!r}")
    if plugin["tools"] != EXPECTED_TOOLS:
        raise RuntimeError(f"expected {EXPECTED_TOOLS} tools, got {plugin['tools']}")
    missing = [name for name in SMOKE_TOOLS if registry.get_entry(name) is None]
    if missing:
        raise RuntimeError(f"missing registered tools: {missing}")
    platform = platform_registry.get("linear_agent")
    if platform is None:
        raise RuntimeError("linear_agent platform was not registered")

    print(
        json.dumps(
            {
                "plugin": plugin["name"],
                "key": plugin["key"],
                "source": plugin["source"],
                "version": plugin["version"],
                "tools": plugin["tools"],
                "platform": platform.name,
                "status": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
