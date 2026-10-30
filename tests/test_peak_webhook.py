"""Regression tests: webhook replay freshness, signature robustness, defensiveness.

Covers three hardening items:

* replay guard — a signed delivery with no timestamp anywhere (neither a
  timestamp header nor the signed ``webhookTimestamp`` body field) must be
  rejected instead of staying valid forever;
* a non-ASCII signature header must be an ordinary invalid signature, not a
  ``TypeError`` that 500s the endpoint;
* ``mutation_policy`` gates must require the boolean ``True``, not any
  non-empty string.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest
from test_linear_agent_adapter import (
    _body,
    _created_payload,
    _make_adapter,
    _signature,
)

from hermes_linear_agent.client import LinearApiError, LinearGraphQLClient
from hermes_linear_agent.webhook import (
    is_stale_body_timestamp,
    verify_linear_signature,
)


def _ms(seconds_offset: float = 0.0) -> str:
    return str(int((time.time() + seconds_offset) * 1000))


# ---------------------------------------------------------------------------
# Replay freshness: a timestamp is mandatory for signed deliveries
# ---------------------------------------------------------------------------


def test_signed_delivery_without_any_timestamp_is_rejected():
    """No header timestamp and no body webhookTimestamp → not verifiable."""
    raw = _body({"action": "created"})
    assert verify_linear_signature({"Linear-Signature": _signature(raw, "secret")}, raw, "secret") is False


def test_signed_delivery_with_body_only_timestamp_missing_is_rejected():
    """An unrelated body field must not stand in for a freshness anchor."""
    raw = _body({"action": "created", "createdAt": "2020-01-01T00:00:00Z"})
    assert verify_linear_signature({"Linear-Signature": _signature(raw, "secret")}, raw, "secret") is False


def test_signed_delivery_with_fresh_header_timestamp_is_accepted():
    raw = _body({"action": "created"})
    assert verify_linear_signature(
        {"Linear-Signature": _signature(raw, "secret"), "Linear-Timestamp": _ms()},
        raw,
        "secret",
    )


def test_signed_delivery_with_stale_header_timestamp_is_rejected():
    raw = _body({"action": "created"})
    assert (
        verify_linear_signature(
            {"Linear-Signature": _signature(raw, "secret"), "Linear-Timestamp": _ms(-3600)},
            raw,
            "secret",
        )
        is False
    )


def test_signed_delivery_with_fresh_body_timestamp_is_accepted():
    raw = _body({"action": "created", "webhookTimestamp": int(time.time() * 1000)})
    assert verify_linear_signature({"Linear-Signature": _signature(raw, "secret")}, raw, "secret")
    assert is_stale_body_timestamp(json.loads(raw), {}) is False


def test_signed_delivery_with_stale_body_timestamp_is_rejected():
    payload = {"action": "created", "webhookTimestamp": int((time.time() - 3600) * 1000)}
    raw = _body(payload)
    # Presence/parseability is satisfied by the body anchor; the window is
    # enforced by the body-timestamp replay guard (the adapter's 400 path).
    assert verify_linear_signature({"Linear-Signature": _signature(raw, "secret")}, raw, "secret")
    assert is_stale_body_timestamp(payload, {}) is True


@pytest.mark.asyncio
async def test_timestampless_signed_webhook_gets_401_from_adapter():
    """End to end: the adapter's rejection path for a bad signature (401)."""
    adapter = _make_adapter()
    raw = _body(_created_payload())

    response, status = await adapter.handle_webhook({"Linear-Signature": _signature(raw, "secret")}, raw)

    assert status == 401
    assert response["error"] == "Invalid signature"


@pytest.mark.asyncio
async def test_fresh_timestamped_signed_webhook_is_accepted_by_adapter():
    adapter = _make_adapter()
    adapter.handle_message = _noop
    payload = _created_payload()
    payload["webhookTimestamp"] = int(time.time() * 1000)
    raw = _body(payload)

    response, status = await adapter.handle_webhook({"Linear-Signature": _signature(raw, "secret")}, raw)

    assert status == 200
    assert response["status"] == "accepted"


@pytest.mark.asyncio
async def test_stale_body_timestamp_signed_webhook_gets_400_from_adapter():
    adapter = _make_adapter()
    payload = _created_payload()
    payload["webhookTimestamp"] = int((time.time() - 3600) * 1000)
    raw = _body(payload)

    response, status = await adapter.handle_webhook({"Linear-Signature": _signature(raw, "secret")}, raw)

    assert status == 400
    assert response["error"] == "Stale webhook delivery"


async def _noop(_event):
    return None


# ---------------------------------------------------------------------------
# Non-ASCII signature headers must not raise
# ---------------------------------------------------------------------------


def test_non_ascii_signature_is_invalid_not_a_typeerror():
    raw = _body({"action": "created", "webhookTimestamp": int(time.time() * 1000)})

    assert verify_linear_signature({"Linear-Signature": "sha256=sigñature"}, raw, "secret") is False


def test_non_ascii_signature_piece_does_not_mask_a_valid_piece():
    """Only the uncomparable piece is skipped; a valid one still matches."""
    raw = _body({"action": "created"})
    digest = hmac.new(b"secret", raw, hashlib.sha256).hexdigest()
    header = f"v1=é {digest}"

    assert verify_linear_signature(
        {"Linear-Signature": header, "Linear-Timestamp": _ms()},
        raw,
        "secret",
    )


@pytest.mark.asyncio
async def test_non_ascii_signature_gets_401_instead_of_500():
    adapter = _make_adapter()
    raw = _body(_created_payload())

    response, status = await adapter.handle_webhook(
        {"Linear-Signature": "sha256=ñ", "Linear-Timestamp": _ms()},
        raw,
    )

    assert status == 401
    assert response["error"] == "Invalid signature"


# ---------------------------------------------------------------------------
# mutation_policy: only the boolean True unlocks a write
# ---------------------------------------------------------------------------


class _RecordingClient(LinearGraphQLClient):
    def __init__(self) -> None:
        super().__init__("token")
        self.executed = False

    async def execute(self, query: str, variables: dict | None = None) -> dict:
        self.executed = True
        return {"issueUpdate": {"success": True, "issue": {"id": "issue-1"}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["true", "false", "yes", 1])
async def test_string_policy_values_do_not_enable_writes(value):
    client = _RecordingClient()

    with pytest.raises(LinearApiError) as excinfo:
        await client.update_issue("issue-1", {"title": "x"}, mutation_policy={"update_issues": value})

    assert "mutation_policy" in str(excinfo.value)
    assert client.executed is False


@pytest.mark.asyncio
async def test_boolean_true_policy_still_enables_writes():
    client = _RecordingClient()

    await client.update_issue("issue-1", {"title": "x"}, mutation_policy={"update_issues": True})

    assert client.executed is True


def test_sender_is_read_from_linear_agent_session_event_shape():
    """P10: AgentSessionEventWebhookPayload names the sender agentActivity.userId
    (prompted) and agentSession.creatorId (created); no top-level actor."""
    from hermes_linear_agent.webhook import extract_context

    issue = {"id": "issue-1", "identifier": "ENG-1", "team": {"id": "team-1"}}
    prompted = {
        "type": "AgentSessionEvent",
        "action": "prompted",
        "organizationId": "org-1",
        "agentSession": {"id": "sess-1", "issue": issue, "creatorId": "creator-1"},
        "agentActivity": {
            "id": "act-1",
            "userId": "tom-1",
            "user": {"id": "tom-1", "name": "Tom"},
            "content": {"type": "prompt", "body": "hello"},
        },
    }
    ctx = extract_context(prompted, {})
    assert ctx.actor_user_id == "tom-1"
    assert ctx.actor_user_name == "Tom"

    created = {
        "type": "AgentSessionEvent",
        "action": "created",
        "organizationId": "org-1",
        "agentSession": {
            "id": "sess-2",
            "issue": issue,
            "creatorId": "tom-1",
            "creator": {"id": "tom-1", "name": "Tom"},
            "comment": {"id": "c-1", "body": "@gilfoyle hi"},
        },
    }
    ctx = extract_context(created, {})
    assert ctx.actor_user_id == "tom-1"
