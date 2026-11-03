"""A turn that arrives while the gateway drains is queued and replayed after restart."""

import asyncio

import pytest
from test_peak_adapter import _FakeLinearClient, _make_adapter, _post, _prompted_payload, _settle

from hermes_linear_agent import adapter as adapter_mod


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _setup():
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client)
    handled = []

    async def capture(event):
        handled.append(event)

    adapter.handle_message = capture
    return adapter, client, handled


async def test_prompted_webhook_while_draining_is_queued_with_a_thought(monkeypatch):
    adapter, client, handled = _setup()
    adapter._latest_agent_session["issue-1"] = "session-1"
    monkeypatch.setattr(adapter_mod, "gateway_draining", lambda: True)

    await _post(adapter, _prompted_payload(body="Keep going."))
    await _settle()

    assert handled == []
    queued, interrupted = adapter._turn_journal.take_pending()
    assert [(e.issue_id, e.issue_key, e.text) for e in queued] == [("issue-1", "LIN-123", "Keep going.")]
    assert interrupted == []
    thoughts = [c for c in client.calls if c[0] == "thought"]
    assert thoughts == [("thought", "session-1", adapter_mod.RESTART_QUEUED_NOTICE)]


async def test_replay_dispatches_queued_text_through_wake(monkeypatch):
    adapter, _client, _handled = _setup()
    adapter._turn_journal.queue("issue-1", "LIN-123", "Keep going.")
    monkeypatch.setattr(adapter_mod, "gateway_ready", lambda: True)
    wakes = []
    monkeypatch.setattr(adapter, "_dispatch_wake", lambda *a: wakes.append(a))

    await adapter._replay_after_restart()

    assert wakes == [("issue-1", "LIN-123", "Message received while the gateway was restarting:\n\nKeep going.")]


async def test_replay_proceeds_after_the_max_wait(monkeypatch):
    adapter, _client, _handled = _setup()
    adapter._turn_journal.queue("issue-1", "LIN-123", "x")
    monkeypatch.setattr(adapter_mod, "gateway_ready", lambda: False)
    monkeypatch.setattr(adapter_mod, "REPLAY_POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(adapter_mod, "REPLAY_MAX_WAIT_S", 0.01)
    wakes = []
    monkeypatch.setattr(adapter, "_dispatch_wake", lambda *a: wakes.append(a))

    await asyncio.wait_for(adapter._replay_after_restart(), 5)

    assert len(wakes) == 1


async def test_runtime_draining_send_posts_nothing_and_requeues():
    adapter, client, _handled = _setup()
    adapter._latest_agent_session["issue-1"] = "session-1"
    adapter._turn_journal.start("issue-1", "LIN-123", "Keep going.")

    result = await adapter.send("issue-1", "⚠️ RuntimeStoreError: runtime_draining")

    assert result.success is True
    assert [c[0] for c in client.calls] == ["thought"]
    queued, interrupted = adapter._turn_journal.take_pending()
    assert [e.text for e in queued] == ["Keep going."]
    assert interrupted == []
