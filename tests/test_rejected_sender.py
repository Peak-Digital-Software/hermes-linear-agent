import pytest
from test_linear_agent_adapter import (
    _body,
    _created_payload,
    _FakeLinearClient,
    _headers,
    _make_adapter,
    _prompted_payload,
)

EXPECTED = "Rejected: Ada (user-1) is not in this agent's allowed users."


def _setup(check, *, extra=None):
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client, extra=extra)
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    if check is not None:
        adapter._authorization_check = check
    return adapter, client, captured


async def _post(adapter, payload):
    raw = _body(payload)
    return await adapter.handle_webhook(_headers(raw, "secret", payload["id"]), raw)


@pytest.mark.asyncio
async def test_rejected_created_posts_single_error_and_does_not_dispatch():
    seen = []

    def check(user_id, chat_type, chat_id, **kwargs):
        seen.append((user_id, chat_type, chat_id, kwargs))
        return False

    adapter, client, captured = _setup(check)
    response, status = await _post(adapter, _created_payload())

    assert status == 200
    assert response == {
        "status": "rejected",
        "reason": "sender not allowed by gateway",
        "delivery_id": "evt-1",
    }
    assert client.calls == [("error", "session-1", EXPECTED)]
    assert captured == []
    assert seen == [("user-1", "thread", "issue-1", {"thread_id": "issue-1"})]


@pytest.mark.asyncio
async def test_rejected_prompted_posts_error():
    adapter, client, captured = _setup(lambda *a, **k: False)
    response, status = await _post(adapter, _prompted_payload())

    assert status == 200
    assert response["status"] == "rejected"
    assert client.calls == [("error", "session-1", EXPECTED)]
    assert captured == []


@pytest.mark.asyncio
async def test_redelivery_does_not_post_twice():
    adapter, client, _ = _setup(lambda *a, **k: False)
    await _post(adapter, _created_payload())
    response, status = await _post(adapter, _created_payload())

    assert status == 200
    assert response["status"] == "duplicate"
    assert [c for c in client.calls if c[0] == "error"] == [("error", "session-1", EXPECTED)]


@pytest.mark.asyncio
async def test_create_error_failure_does_not_raise():
    adapter, client, captured = _setup(lambda *a, **k: False)

    async def boom(*_a, **_k):
        raise RuntimeError("linear down")

    client.create_error = boom
    response, status = await _post(adapter, _created_payload())

    assert status == 200
    assert response["status"] == "rejected"
    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize("check", [lambda *a, **k: True, None])
async def test_allowed_or_absent_callback_dispatches(check):
    adapter, client, captured = _setup(check)
    response, status = await _post(adapter, _created_payload())

    assert status == 200
    assert response["status"] == "accepted"
    assert len(captured) == 1
    assert not [c for c in client.calls if c[0] == "error"]
