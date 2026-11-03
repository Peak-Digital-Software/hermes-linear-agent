from __future__ import annotations

import pytest

from hermes_linear_agent import go_gate


@pytest.mark.parametrize(
    "text",
    [
        "Plan is ready.\n\nReply go to proceed.",
        "Plan is ready.\n\nreply `go` and I will start",
        'Done.\n\nSay "ship" when ready.',
        "Done.\n\nFile it?",
        "Done.\n\nWant me to open the PR?",
        "Done.\n\nShall I continue?",
        "Done.\n\nSHOULD I merge it?",
    ],
)
def test_last_paragraph_patterns_match(text):
    source = go_gate.match(text)
    assert source in go_gate.GO_GATE_PATTERNS


def test_match_returns_the_pattern_source():
    assert go_gate.match("x\n\nshall i go on") == r"\bshall i\b"


def test_only_the_last_paragraph_counts():
    assert go_gate.match("Want me to do it?\n\nIt is done. All merged.") is None


def test_no_match_mid_text():
    assert go_gate.match("Should I worry? Shall I panic?\n\nNo, we are fine.") is None
    assert go_gate.match("I said that before.\n\nNothing pending.") is None


def test_single_paragraph_is_the_last_paragraph():
    assert go_gate.match("Want me to continue?") is not None


def test_empty_text_does_not_match():
    assert go_gate.match("") is None
    assert go_gate.match("   \n\n  ") is None


def test_scope_for():
    assert go_gate.scope_for({"state": {"name": "Working", "type": "started"}}) == "gated"
    assert go_gate.scope_for({"state": {"name": "Ready to Merge", "type": "started"}}) == "exempt"
    assert go_gate.scope_for({"state": {"name": "Triage", "type": "triage"}, "delegate": None}) == "exempt"
    assert go_gate.scope_for({"state": {"name": "Triage", "type": "triage"}, "delegate": {"id": "a"}}) == "gated"
    assert go_gate.scope_for(None) == "unknown"
