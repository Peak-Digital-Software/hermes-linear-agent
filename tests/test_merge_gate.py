from __future__ import annotations

import json

import pytest
from test_linear_agent_adapter import _FakeLinearClient, _make_adapter

from hermes_linear_agent import merge_gate

_MERGE_TEXT = "Verified.\n\nReady for you to merge #12."


@pytest.mark.parametrize(
    "text",
    [
        "Done.\n\nPlease merge it.",
        "Done.\n\nMerge #12 when ready.",
        "Done.\n\nMerge [repo#12] please.",
        "Done.\n\nReady for you to merge.",
        "Done.\n\nNext: merge.",
    ],
)
def test_merge_asks_match(text):
    assert merge_gate.match(text)


@pytest.mark.parametrize(
    "text",
    [
        "I merged #12 and deployed.",
        "Done.\n\nPR #12 was merged by Oracle.",
        "Please merge it.\n\nAll checks are green.",
        "",
    ],
)
def test_reports_and_earlier_paragraphs_do_not_match(text):
    assert merge_gate.match(text) is None


def test_allowed_only_for_ready_to_merge_or_done():
    assert merge_gate.allowed({"state": {"name": "Ready to Merge"}})
    assert merge_gate.allowed({"state": {"name": "Done"}})
    assert not merge_gate.allowed({"state": {"name": "Reviewing"}})
    assert not merge_gate.allowed({})
    assert not merge_gate.allowed(None)


def _hits(tmp_path):
    path = tmp_path / "merge-gate-hits.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


async def _send(tmp_path, monkeypatch, issue):
    monkeypatch.setenv("PEAK_AGENT_STATE_DIR", str(tmp_path))
    client = _FakeLinearClient()
    client.issue = issue
    adapter = _make_adapter(client=client)
    adapter._latest_agent_session["issue-1"] = "session-1"
    result = await adapter.send("issue-1", _MERGE_TEXT, metadata={})
    assert result.success is True
    return client


@pytest.mark.asyncio
async def test_merge_ask_in_reviewing_posts_a_thought_and_logs(tmp_path, monkeypatch):
    client = await _send(tmp_path, monkeypatch, {"state": {"name": "Reviewing", "type": "started"}})

    assert [c[0] for c in client.calls] == ["get_issue", "thought"]
    hits = _hits(tmp_path)
    assert [(h["posted_as"], h["state"]) for h in hits] == [("thought", "Reviewing")]
    assert hits[0]["session"] == "session-1"


@pytest.mark.asyncio
async def test_merge_ask_in_ready_to_merge_posts_a_response(tmp_path, monkeypatch):
    client = await _send(tmp_path, monkeypatch, {"state": {"name": "Ready to Merge", "type": "started"}})

    assert [c[0] for c in client.calls] == ["get_issue", "response"]
    assert _hits(tmp_path) == []


@pytest.mark.asyncio
async def test_merge_gate_fails_open_when_get_issue_raises(tmp_path, monkeypatch):
    client = await _send(tmp_path, monkeypatch, RuntimeError("linear down"))

    assert client.calls[-1][0] == "response"
    assert _hits(tmp_path) == []


@pytest.mark.asyncio
async def test_past_tense_report_makes_no_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAK_AGENT_STATE_DIR", str(tmp_path))
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client)
    adapter._latest_agent_session["issue-1"] = "session-1"

    await adapter.send("issue-1", "Oracle merged #12.", metadata={})

    assert [c[0] for c in client.calls] == ["response"]


@pytest.mark.asyncio
async def test_go_and_merge_hit_fetch_the_issue_once(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAK_AGENT_STATE_DIR", str(tmp_path))
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client)
    adapter._latest_agent_session["issue-1"] = "session-1"

    await adapter.send("issue-1", "Done.\n\nShould I merge it?", metadata={})

    assert [c[0] for c in client.calls].count("get_issue") == 1
    assert client.calls[-1][0] == "thought"
