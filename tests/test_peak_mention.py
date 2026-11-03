"""A peer agent's issueCommentMention becomes a wake turn, capped per issue."""

import json
from pathlib import Path

import pytest
from test_peak_adapter import _FakeLinearClient, _make_adapter, _post

from hermes_linear_agent import adapter as adapter_mod

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "issue_comment_mention.json").read_text())
ISSUE_ID = FIXTURE["notification"]["issueId"]
ACTOR_ID = FIXTURE["notification"]["actorId"]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_mod, "gateway_draining", lambda: False)


def _setup(*, peers=(ACTOR_ID,), session=True, cap=None):
    client = _FakeLinearClient()
    extra = {"app_user_id": FIXTURE["appUserId"], "peer_app_user_ids": list(peers)}
    if cap is not None:
        extra["max_agent_handoffs"] = cap
    adapter = _make_adapter(extra=extra, client=client)
    if session:
        adapter._latest_agent_session[ISSUE_ID] = "session-1"
    wakes = []
    adapter._dispatch_wake = lambda *a: wakes.append(a)
    return adapter, client, wakes


def _payload(delivery="mention-1", notification_id=None):
    payload = json.loads(json.dumps(FIXTURE))
    payload["id"] = delivery
    if notification_id:
        payload["notification"]["id"] = notification_id
    return payload


async def test_peer_mention_dispatches_one_wake_with_the_comment_body():
    adapter, _client, wakes = _setup()

    body, status = await _post(adapter, _payload())

    assert status == 200 and body == {"status": "accepted"}
    assert len(wakes) == 1
    issue_id, key, text = wakes[0]
    assert (issue_id, key) == (ISSUE_ID, "ENG-2324")
    assert text.startswith("Baxter mentioned you in a comment on ENG-2324:\n\n")
    assert FIXTURE["notification"]["comment"]["body"] in text


async def test_same_delivery_again_is_a_duplicate():
    adapter, _client, wakes = _setup()
    await _post(adapter, _payload())

    body, status = await _post(adapter, _payload())

    assert status == 200 and body == {"status": "duplicate"}
    assert len(wakes) == 1


async def test_non_peer_actor_is_ignored():
    adapter, _client, wakes = _setup(peers=("someone-else",))

    body, status = await _post(adapter, _payload())

    assert status == 200 and body["status"] == "ignored"
    assert wakes == []


async def test_no_session_is_ignored():
    adapter, _client, wakes = _setup(session=False)

    body, status = await _post(adapter, _payload())

    assert status == 200 and body == {"status": "ignored", "reason": "no session"}
    assert wakes == []


async def test_past_the_cap_posts_a_refusal_and_no_wake():
    adapter, client, wakes = _setup(cap=1)
    await _post(adapter, _payload("m1", "n1"))

    body, status = await _post(adapter, _payload("m2", "n2"))

    assert status == 200 and body == {"status": "refused", "reason": "handoff limit reached"}
    assert len(wakes) == 1
    refusals = [c for c in client.calls if c[0] == "response"]
    assert len(refusals) == 1 and "Handoff limit reached: 2" in refusals[0][2]
