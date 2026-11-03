"""ENG-2408 — resolve the delegating user as sender; reject an unresolved one visibly.

The live OPS-38 delivery is the fixture: filed by this app, no assignee, the
app set itself as delegate. Nothing in the payload names a human.
"""

import json
from pathlib import Path

import pytest
from test_peak_adapter import (
    _body,
    _capture_events,
    _FakeLinearClient,
    _headers,
    _make_adapter,
    _post,
    _prompted_payload,
    _settle,
)

from hermes_linear_agent.adapter import NO_SENDER_ERROR
from hermes_linear_agent.webhook import extract_context

_BAXTER = "9c7f2bb0-74a5-47d3-bbeb-ad5caa0fb037"
_TOM = "d9c67fb5-7038-4b32-a266-0e8a1c379be4"
_GILFOYLE = "21d33d4d-d1f5-4348-9dfb-29e46df5156a"
_ISSUE = "d69a3f47-862f-4f53-bb35-0e3a61551612"
_SESSION = "cc1db682-075a-4994-931e-04ed86ff4eac"


class _Client(_FakeLinearClient):
    def __init__(self):
        super().__init__()
        self.delegators = {}
        self.delegator_error = None

    async def get_issue_delegator_user_id(self, issue_id, delegate_id):
        self.calls.append(("delegator", issue_id, delegate_id))
        if self.delegator_error:
            raise self.delegator_error
        return self.delegators.get(issue_id, "")


def _payload():
    path = Path(__file__).parent / "fixtures" / "agent_session_created_self_delegated.json"
    return json.loads(path.read_text())


def _adapter(client):
    client.issue_creators[_ISSUE] = ""
    client.issues[_ISSUE] = {"id": _ISSUE}
    return _make_adapter(
        client=client,
        extra={
            "app_user_id": _BAXTER,
            "allow_all_users": False,
            "allowed_users": [_TOM],
            "peer_app_user_ids": [_GILFOYLE],
        },
    )


async def _post_fixture(adapter, delivery_id="evt-ops38"):
    raw = _body(_payload())
    return await adapter.handle_webhook(_headers(raw, delivery_id=delivery_id), raw)


def _errors(client):
    return [call for call in client.calls if call[0] == "error"]


@pytest.mark.asyncio
async def test_self_delegation_is_rejected_with_an_error_activity():
    client = _Client()
    client.delegators[_ISSUE] = _BAXTER
    adapter = _adapter(client)
    events = await _capture_events(adapter)

    response, status = await _post_fixture(adapter)
    await _settle()

    assert status == 200
    assert response["status"] == "rejected"
    assert _errors(client) == [("error", _SESSION, NO_SENDER_ERROR)]
    assert events == []


@pytest.mark.asyncio
async def test_redelivery_is_a_duplicate_and_posts_one_error():
    client = _Client()
    client.delegators[_ISSUE] = _BAXTER
    adapter = _adapter(client)
    events = await _capture_events(adapter)

    first, _ = await _post_fixture(adapter)
    second, status = await _post_fixture(adapter)
    await _settle()

    assert first["status"] == "rejected"
    assert status == 200
    assert second["status"] == "duplicate"
    assert len(_errors(client)) == 1
    assert events == []


@pytest.mark.asyncio
async def test_a_human_delegator_is_the_sender():
    client = _Client()
    client.delegators[_ISSUE] = _TOM
    adapter = _adapter(client)
    events = await _capture_events(adapter)

    context = await adapter._resolve_sender(extract_context(_payload(), {}))
    assert context.actor_user_id == _TOM
    assert context.sender_source == "issue_delegator"

    _, status = await _post_fixture(adapter)
    await _settle()

    assert status == 200
    assert [event.source.user_id for event in events] == [_TOM]
    assert _errors(client) == []


@pytest.mark.asyncio
async def test_a_peer_app_delegator_is_dispatched_as_the_peer():
    client = _Client()
    client.delegators[_ISSUE] = _GILFOYLE
    adapter = _adapter(client)
    events = await _capture_events(adapter)

    _, status = await _post_fixture(adapter)
    await _settle()

    assert status == 200
    assert [event.source.user_id for event in events] == [_GILFOYLE]
    assert _errors(client) == []


@pytest.mark.asyncio
async def test_a_failing_delegator_lookup_is_rejected_visibly():
    client = _Client()
    client.delegator_error = RuntimeError("linear down")
    adapter = _adapter(client)
    events = await _capture_events(adapter)

    response, status = await _post_fixture(adapter)
    await _settle()

    assert status == 200
    assert response["status"] == "rejected"
    assert _errors(client) == [("error", _SESSION, NO_SENDER_ERROR)]
    assert events == []


@pytest.mark.asyncio
async def test_self_echo_stays_a_silent_403():
    client = _Client()
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": _BAXTER, "allow_all_users": False, "allowed_users": [_TOM]},
    )
    events = await _capture_events(adapter)

    payload = _prompted_payload()
    payload["agentActivity"]["actor"] = {"id": _BAXTER, "name": "Baxter"}
    _, status = await _post(adapter, payload, delivery_id="evt-echo")
    await _settle()

    assert status == 403
    assert events == []
    assert _errors(client) == []
