"""P17: the Linear platform hint only advertises ``linear_agent_*`` when that toolset is on.

``PlatformEntry.platform_hint`` is a plain string decided once at ``register()``
(``~/.hermes/hermes-agent/gateway/platform_registry.py:88``), so the gate is:

* ``build_platform_hint(True)``  -> byte-identical to the pre-P17 hint,
* ``build_platform_hint(False)`` -> the same text with only the toolset paragraph removed,
* ``linear_agent_toolset_enabled`` -> the inlined rule of
  ``hermes_cli/tools_config.py:526-534`` (``_enabled_plugin_toolsets``) for platform and
  toolset ``linear_agent``, failing safe (False) on anything unexpected.
"""

import pytest

from hermes_linear_agent import adapter as adapter_module
from hermes_linear_agent import tools as linear_agent_tools
from hermes_linear_agent.adapter import (
    LINEAR_AGENT_TOOLSET_HINT,
    build_platform_hint,
    linear_agent_toolset_enabled,
    register,
)

# The sentence P18 adds to the toolset paragraph: the model is told to post an
# authentication elicitation instead of pasting a connect URL as plain text.
AUTH_HINT_SENTENCE = (
    "If a tool returns an authentication or connect URL that the user has to open, call "
    "linear_agent_request_auth with that URL (pass the Initiating user ID from the prompt as "
    "user_id) as the last action of the turn, then stop. Do not paste the URL as plain text. "
    "When the session resumes, your first activity should be a thought."
)

# The full hint as pinned after P17, plus the P18 auth sentence inside the
# toolset paragraph (the paragraph is still the only part gated on the toolset).
ORIGIN_MAIN_HINT = (
    'You are responding in a Linear Agent Session. Your final response text is delivered directly '
    "into the session's Linear thread — put the ACTUAL content of your reply there, never a narration "
    "like 'posted X in the thread'. Do NOT use linear_agent_create_comment to talk to the user in the "
    'CURRENT session; that duplicates the conversation and buries your reply. Comment tools are for '
    "OTHER issues, when the user explicitly asks for a comment, or when a 'Reply in the source "
    "thread' instruction is present in your prompt (then follow it exactly). To ask the user "
    'something, ask directly in your response (or via your clarify tool) — the session waits for '
    'their reply. For ALL Linear reads and writes, always prefer the linear_agent_* toolset over the '
    'generic mcp_linear_* tools — linear_agent_* tools authenticate as the Linear Agent, so changes '
    "are attributed to the agent in Linear's history. Writes: linear_agent_update_issue, "
    'linear_agent_create_comment, linear_agent_create_issue, linear_agent_create_project, '
    'linear_agent_update_project, linear_agent_create_project_update, and linear_agent_save_* tools '
    'for documents, initiatives, releases, milestones, status updates, and customer needs. Reads: '
    'linear_agent_list_* / linear_agent_get_* tools for teams, issues, projects, cycles, users, '
    'labels, comments, attachments, customers, initiatives, and releases. Write tools are gated by '
    "the operator's mutation_policy and may be disabled; if a write is rejected, report the policy "
    "message rather than retrying. If a tool returns an authentication or connect URL that the user "
    "has to open, call linear_agent_request_auth with that URL (pass the Initiating user ID from "
    "the prompt as user_id) as the last action of the turn, then stop. Do not paste the URL as "
    "plain text. When the session resumes, your first activity should be a thought. "
    "When changing an issue's status you may pass the state NAME (e.g. "
    "state: 'Done') — it is resolved to the correct stateId automatically; likewise pass priority as "
    "a NAME (e.g. priority: 'Low') — Linear's numeric scale is 0=None, 1=Urgent, 2=High, 3=Medium, "
    "4=Low, so a guessed number silently sets the wrong value. Never set yourself as an issue's "
    'delegate or assignee unless the user explicitly asks you to. When your work produces an external '
    'artifact (a PR, document, or dashboard), attach it with linear_agent_set_session_links using the '
    'Session ID from your prompt. For multi-step work, mirror your todo plan into '
    "linear_agent_update_plan (send the FULL step list every time, updating each step's status as it "
    'starts and finishes) so Linear renders live progress. Never write tool-call syntax (e.g. '
    '<tool_code>) into your response text; always emit real tool calls.'
)


class _PluginContext:
    def __init__(self):
        self.registered = None

    def register_platform(self, **kwargs):
        self.registered = kwargs

    def register_tool(self, **kwargs):
        return None


def test_hint_with_the_toolset_on_is_the_origin_main_string():
    assert build_platform_hint(True) == ORIGIN_MAIN_HINT


def test_auth_sentence_is_in_the_hint_when_the_toolset_is_on():
    assert AUTH_HINT_SENTENCE in LINEAR_AGENT_TOOLSET_HINT
    assert AUTH_HINT_SENTENCE in build_platform_hint(True)


def test_auth_sentence_is_absent_when_the_toolset_is_off():
    hint = build_platform_hint(False)

    assert AUTH_HINT_SENTENCE not in hint
    assert "linear_agent_request_auth" not in hint


def test_hint_with_the_toolset_off_drops_only_that_paragraph():
    hint = build_platform_hint(False)

    assert hint == ORIGIN_MAIN_HINT.replace(LINEAR_AGENT_TOOLSET_HINT, "")
    assert "linear_agent_*" not in hint
    assert "mcp_linear_*" not in hint
    assert "For ALL Linear reads" not in hint
    # Everything outside the paragraph survives verbatim.
    assert "You are responding in a Linear Agent Session." in hint
    assert "When changing an issue's status you may pass the state NAME" in hint
    assert "Never write tool call syntax" not in hint  # sanity: not a phrase from the real hint
    assert "Never write tool-call syntax" in hint
    # No stitched-together leftovers.
    assert "\n\n" not in hint
    assert "  " not in hint
    assert not hint.endswith(" ")


@pytest.mark.parametrize("enabled", [True, False])
def test_register_uses_the_gate_for_the_platform_hint(monkeypatch, enabled):
    monkeypatch.setattr(linear_agent_tools, "register_tools_with_context", lambda ctx: None)
    monkeypatch.setattr(adapter_module, "linear_agent_toolset_enabled", lambda config=None: enabled)

    ctx = _PluginContext()
    register(ctx)

    assert ctx.registered["platform_hint"] == build_platform_hint(enabled)


# Config shapes: a saved platform list that omits `linear_agent` (the shape seen
# before the toolset was enabled) resolves to False; a list that names it resolves
# to True, no matter what `known_plugin_toolsets` records.


def test_gilfoyle_shaped_config_resolves_to_false():
    config = {
        "platform_toolsets": {"linear_agent": ["terminal", "file", "web"]},
        "known_plugin_toolsets": {"linear_agent": ["linear_agent"]},
    }

    assert linear_agent_toolset_enabled(config) is False


def test_toolset_listed_for_the_platform_resolves_to_true():
    config = {
        "platform_toolsets": {"linear_agent": ["terminal", "linear_agent"]},
        "known_plugin_toolsets": {"linear_agent": ["linear_agent"]},
    }

    assert linear_agent_toolset_enabled(config) is True


def test_absent_platform_list_keeps_the_plugin_toolset_on():
    """No saved list: plugin toolsets are on unless default-off or known-and-absent."""
    assert linear_agent_toolset_enabled({}) is True
    assert linear_agent_toolset_enabled({"platform_toolsets": {}, "known_plugin_toolsets": {}}) is True


def test_absent_platform_list_with_a_known_entry_resolves_to_false():
    config = {"platform_toolsets": {}, "known_plugin_toolsets": {"linear_agent": ["linear_agent"]}}

    assert linear_agent_toolset_enabled(config) is False


def test_load_config_failure_fails_safe(monkeypatch):
    import hermes_cli.config as core_config

    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(core_config, "load_config", _boom)

    assert linear_agent_toolset_enabled() is False


@pytest.mark.parametrize("config", [[], "linear_agent"])
def test_unusable_config_fails_safe(config):
    """A non-dict config never enables the paragraph (and never raises)."""
    assert linear_agent_toolset_enabled(config) is False
