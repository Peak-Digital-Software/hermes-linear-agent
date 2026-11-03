"""ENG-2327: exec approvals as Linear elicitations, resolved by free-text replies."""

import asyncio
import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from tools import approval

from hermes_linear_agent.adapter import APPROVAL_REPLY_LINE, LinearAgentAdapter
from hermes_linear_agent.webhook import extract_context


class _Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.responses = []
        self.thoughts = []

    async def create_response(self, agent_session_id, body, *, response_type="response", **kwargs):
        if self.fail:
            raise RuntimeError("linear down")
        self.responses.append((agent_session_id, body, response_type))
        return {"id": "elicit-1"}

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        self.thoughts.append((agent_session_id, body))
        return {"id": "thought-1"}

    async def create_error(self, agent_session_id, body):
        return {"id": "error-1"}

    async def list_issue_agent_sessions(self, issue_id, limit=50):
        return []


def _adapter(client):
    return LinearAgentAdapter(
        PlatformConfig(
            enabled=True,
            extra={"access_token": "t", "webhook_secret": "secret", "allow_all_users": True},
        ),
        client=client,
    )


def _prompt(text="Command: `rm -rf /tmp/x`\nReason: recursive delete", chat_id="issue-1", metadata=None):
    return SimpleNamespace(chat_id=chat_id, metadata=metadata, text=text)


def _payload(text, event_id="evt-1"):
    return {
        "action": "prompted",
        "id": event_id,
        "agentSession": {
            "id": "session-1",
            "workspaceId": "workspace-1",
            "issue": {"id": "issue-1", "identifier": "LIN-1", "title": "T", "team": {"id": "team-1"}},
        },
        "agentActivity": {"id": f"act-{event_id}", "body": text, "actor": {"id": "user-1", "name": "Ada"}},
    }


def _raw(payload):
    return json.dumps(payload, separators=(",", ":")).encode()


def _headers(raw, delivery_id):
    return {
        "Linear-Signature": "sha256=" + hmac.new(b"secret", raw, hashlib.sha256).hexdigest(),
        "Linear-Timestamp": str(int(time.time() * 1000)),
        "Linear-Delivery-Id": delivery_id,
    }


# ---- A: the prompt ---------------------------------------------------------


def test_adapter_renders_exec_approvals_natively():
    assert LinearAgentAdapter.supports_exec_approval_buttons() is True


async def test_prompt_posts_one_elicitation_with_command_and_reply_line():
    client = _Client()
    adapter = _adapter(client)
    adapter._reply_target = lambda chat_id, metadata=None: ("session-1", "issue-1")
    prompt = _prompt()

    result = await adapter._send_exec_approval_prompt(prompt)

    assert result.success is True
    assert result.message_id == "elicit-1"
    assert client.responses == [("session-1", f"{prompt.text}\n\n{APPROVAL_REPLY_LINE}", "elicitation")]
    assert APPROVAL_REPLY_LINE == "Reply `approve`, `always` or `deny`."


async def test_prompt_is_truncated_to_the_message_limit():
    client = _Client()
    adapter = _adapter(client)
    adapter._reply_target = lambda chat_id, metadata=None: ("session-1", "issue-1")

    result = await adapter._send_exec_approval_prompt(_prompt(text="x" * (adapter.MAX_MESSAGE_LENGTH + 500)))

    assert result.success is True
    body = client.responses[0][1]
    assert len(body) <= adapter.MAX_MESSAGE_LENGTH
    assert body.endswith(APPROVAL_REPLY_LINE)


async def test_prompt_fails_without_a_session():
    client = _Client()
    adapter = _adapter(client)

    async def no_session(chat_id, metadata=None):
        return "", ""

    adapter._resolve_reply_session = no_session
    result = await adapter._send_exec_approval_prompt(_prompt())

    assert isinstance(result, SendResult) and result.success is False
    assert client.responses == []


async def test_prompt_fails_when_the_client_raises():
    adapter = _adapter(_Client(fail=True))
    adapter._reply_target = lambda chat_id, metadata=None: ("session-1", "issue-1")

    result = await adapter._send_exec_approval_prompt(_prompt())

    assert result.success is False
    assert "linear down" in result.error


async def test_core_button_branch_side_calls_do_not_raise():
    """Core's button branch edits the card on timeout; the base edit must report failure, not raise."""
    adapter = _adapter(_Client())
    result = await adapter.edit_message("issue-1", "elicit-1", "timed out")
    assert result.success is False


# ---- B: free-text replies --------------------------------------------------


@pytest.fixture
def resolver(monkeypatch):
    calls = []
    state = {"blocking": True}
    monkeypatch.setattr(approval, "has_blocking_approval", lambda key: state["blocking"])

    def resolve(key, choice, *args, **kwargs):
        calls.append((key, choice))
        return 1

    monkeypatch.setattr(approval, "resolve_gateway_approval", resolve)
    return SimpleNamespace(calls=calls, state=state)


async def _dispatch(adapter, text):
    handled = []

    async def handle_message(event):
        handled.append(event.text)

    adapter.handle_message = handle_message
    payload = _payload(text)
    context = extract_context(payload, {})
    await adapter._dispatch_linear_message(context, context.user_request, payload)
    return handled


@pytest.mark.parametrize(
    ("text", "choice"),
    [
        ("Approve cutover", "once"),
        ("approved.", "once"),
        ("Once", "once"),
        ("session please", "session"),
        ("deny, wrong host", "deny"),
        ("Reject: no", "deny"),
        ("always", "always"),
        ("  **Always** do this", "always"),
    ],
)
async def test_reply_resolves_a_blocking_approval_and_is_not_dispatched(resolver, text, choice):
    client = _Client()
    adapter = _adapter(client)

    handled = await _dispatch(adapter, text)

    assert handled == []
    assert [c for _, c in resolver.calls] == [choice]
    assert client.thoughts == [("session-1", "Denied" if choice == "deny" else f"Approved: {choice}")]


@pytest.mark.parametrize("text", ["yes", "no", "approval looks fine?", "can you approve this later", "Disapprove"])
async def test_other_replies_dispatch_normally(resolver, text):
    adapter = _adapter(_Client())

    handled = await _dispatch(adapter, text)

    assert handled == [text]
    assert resolver.calls == []


async def test_approve_without_a_blocking_approval_dispatches(resolver):
    resolver.state["blocking"] = False
    adapter = _adapter(_Client())

    handled = await _dispatch(adapter, "approve")

    assert handled == ["approve"]
    assert resolver.calls == []


async def test_resolver_error_falls_through_to_dispatch(monkeypatch):
    def boom(key):
        raise RuntimeError("store down")

    monkeypatch.setattr(approval, "has_blocking_approval", boom)
    adapter = _adapter(_Client())

    assert await _dispatch(adapter, "approve") == ["approve"]


async def test_reply_key_matches_the_key_the_turn_registered_under(monkeypatch):
    """The key used to look up approvals must be the one real handle_message files the turn under.

    A mismatch would make the free-text path silently no-op. Turn 1 runs through the real
    ``handle_message`` and records its authoritative session key; a real approval queue entry is
    then seeded under that key and turn 2 ("Approve cutover") must find and resolve it.
    """
    adapter = _adapter(_Client())
    turn_keys = []

    async def capture_background(event, session_key):
        turn_keys.append(session_key)

    async def message_handler(event):
        return None

    adapter._message_handler = message_handler
    adapter._process_message_background = capture_background

    raw = _raw(_payload("check the log", "evt-1"))
    _, status = await adapter.handle_webhook(_headers(raw, "evt-1"), raw)
    await asyncio.sleep(0)
    assert status == 200 and len(turn_keys) == 1

    resolved = []
    monkeypatch.setattr(
        approval, "resolve_gateway_approval", lambda key, choice, *a, **k: resolved.append((key, choice)) or 1
    )
    approval._gateway_queues[turn_keys[0]] = [object()]  # real has_blocking_approval reads this
    try:
        raw = _raw(_payload("Approve cutover", "evt-2"))
        _, status = await adapter.handle_webhook(_headers(raw, "evt-2"), raw)
        await asyncio.sleep(0)
    finally:
        approval._gateway_queues.pop(turn_keys[0], None)

    assert status == 200
    assert resolved == [(turn_keys[0], "once")]
    assert len(turn_keys) == 1, "the approval reply must not reach core as a turn"
