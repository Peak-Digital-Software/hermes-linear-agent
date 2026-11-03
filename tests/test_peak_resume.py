"""A turn interrupted by a restart is resumed once; a finished one is not."""

import pytest
from test_peak_adapter import _FakeLinearClient, _make_adapter, _post, _prompted_payload, _settle

from hermes_linear_agent import adapter as adapter_mod

RESUME = "Resume after gateway restart"


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_mod, "gateway_ready", lambda: True)
    monkeypatch.setattr(adapter_mod, "gateway_draining", lambda: False)


def _adapter():
    adapter = _make_adapter(client=_FakeLinearClient())

    async def capture(event):
        pass

    adapter.handle_message = capture
    adapter._latest_agent_session["issue-1"] = "session-1"
    return adapter


async def _replay(monkeypatch):
    fresh = _adapter()
    wakes = []
    monkeypatch.setattr(fresh, "_dispatch_wake", lambda *a: wakes.append(a))
    await fresh._replay_after_restart()
    return wakes


async def test_turn_without_final_send_is_resumed_exactly_once(monkeypatch):
    first = _adapter()
    await _post(first, _prompted_payload())
    await _settle()

    assert await _replay(monkeypatch) == [("issue-1", "LIN-123", RESUME)]
    assert await _replay(monkeypatch) == []


async def test_turn_with_final_send_is_not_resumed(monkeypatch):
    first = _adapter()
    await _post(first, _prompted_payload())
    await _settle()
    assert (await first.send("issue-1", "Done.")).success is True

    assert await _replay(monkeypatch) == []


async def test_interim_send_does_not_clear_the_mark(monkeypatch):
    first = _adapter()
    await _post(first, _prompted_payload())
    await _settle()
    await first.send("issue-1", "Gateway shutting down", metadata={"_interim_send": True})

    assert await _replay(monkeypatch) == [("issue-1", "LIN-123", RESUME)]


async def test_error_activity_clears_the_mark(monkeypatch):
    first = _adapter()
    await _post(first, _prompted_payload())
    await _settle()
    await first.send_error_activity("issue-1", "boom")

    assert await _replay(monkeypatch) == []
