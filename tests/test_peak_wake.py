"""Tests for the signed loopback wake endpoint (POST /internal/wake).

The endpoint is a local control-plane hook: authorized by an HMAC over the
timestamp and the raw body, never by the Linear user allowlist, and routed
outside the public webhook path so a tunnel cannot reach it.
"""

import asyncio
import hashlib
import hmac
import json
import time

from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig

from hermes_linear_agent.adapter import DEFAULT_WAKE_PATH, LinearAgentAdapter
from hermes_linear_agent.webhook import LinearWebhookContext

WAKE_SECRET = "wake-secret"
ISSUE_ID = "8f14e45f-ceea-467f-a1b2-3c4d5e6f7081"
ISSUE_KEY = "ENG-2218"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeLinearClient:
    """Just enough client for adapter construction; turns are recorded instead."""

    configured = True

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        return {"id": "thought-1"}

    async def create_response(self, agent_session_id, body, **kwargs):
        return {"id": "response-1"}

    async def create_error(self, agent_session_id, body):
        return {"id": "error-1"}


class _Recorder:
    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)


def _make_adapter(*, wake_secret=WAKE_SECRET, extra=None):
    merged = {
        "access_token": "linear-token",
        "webhook_secret": "secret",
        "allow_all_users": True,
        "app_user_id": "app-1",
        "workspace_id": "workspace-1",
    }
    if wake_secret:
        merged["wake_secret"] = wake_secret
    if extra:
        merged.update(extra)
    return LinearAgentAdapter(PlatformConfig(enabled=True, extra=merged), client=_FakeLinearClient())


def _install_recorder(adapter):
    recorder = _Recorder()
    adapter.handle_message = recorder
    return recorder


def _body(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _signed(raw, *, secret=WAKE_SECRET, timestamp=None):
    stamp = str(int(time.time()) if timestamp is None else timestamp)
    signature = hmac.new(secret.encode("utf-8"), f"{stamp}.".encode() + raw, hashlib.sha256).hexdigest()
    return {"X-Wake-Timestamp": stamp, "X-Wake-Signature": signature}


async def _settle():
    """Let the background dispatch task reach the recorder."""
    for _ in range(20):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# 1. happy path
# ---------------------------------------------------------------------------


async def test_signed_wake_dispatches_one_internal_turn(monkeypatch):
    # The secret is read through the profile-scoped env helper, like the
    # webhook secret.
    monkeypatch.setenv("LINEAR_AGENT_WAKE_SECRET", WAKE_SECRET)
    adapter = _make_adapter(wake_secret=None)
    recorder = _install_recorder(adapter)
    raw = _body({"issue_id": ISSUE_ID, "issue_key": ISSUE_KEY, "text": "check the deploy"})

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(DEFAULT_WAKE_PATH, data=raw, headers=_signed(raw))
        assert response.status == 202
        assert await response.json() == {"status": "dispatched"}

    await _settle()
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.internal is True
    assert event.text == "[peak-agent wake] check the deploy"
    assert event.source.chat_id == ISSUE_ID
    assert event.source.thread_id == ISSUE_ID
    assert event.source.chat_type == "thread"
    assert event.source.guild_id == "workspace-1"
    # Never a Linear user: the wake turn belongs to this app.
    assert event.source.user_id == "app-1"
    assert event.source.chat_name == ISSUE_KEY

    context = LinearWebhookContext(
        action="prompted",
        delivery_id="wake",
        agent_session_id="",
        workspace_id="workspace-1",
        issue_id=ISSUE_ID,
        issue_identifier=ISSUE_KEY,
    )
    assert adapter._source_session_key(event.source) == adapter._session_key_for(context)


# ---------------------------------------------------------------------------
# 2-3. signature and freshness
# ---------------------------------------------------------------------------


async def test_bad_signature_is_rejected_without_dispatch():
    adapter = _make_adapter()
    recorder = _install_recorder(adapter)
    raw = _body({"issue_id": ISSUE_ID, "text": "check the deploy"})
    headers = _signed(raw)
    headers["X-Wake-Signature"] = "0" * 64

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(DEFAULT_WAKE_PATH, data=raw, headers=headers)
        assert response.status == 401

    await _settle()
    assert recorder.events == []


async def test_stale_timestamp_is_rejected_without_dispatch():
    adapter = _make_adapter()
    recorder = _install_recorder(adapter)
    raw = _body({"issue_id": ISSUE_ID, "text": "check the deploy"})
    headers = _signed(raw, timestamp=int(time.time()) - 301)

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(DEFAULT_WAKE_PATH, data=raw, headers=headers)
        assert response.status == 401

    await _settle()
    assert recorder.events == []


# ---------------------------------------------------------------------------
# 4. fail closed when disabled
# ---------------------------------------------------------------------------


async def test_unset_secret_disables_the_endpoint():
    # No wake_secret in extra and no env var (conftest strips LINEAR_*).
    adapter = _make_adapter(wake_secret=None)
    recorder = _install_recorder(adapter)
    raw = _body({"issue_id": ISSUE_ID, "text": "check the deploy"})
    # Even a request signed with the WEBHOOK secret must not be accepted.
    headers = _signed(raw, secret="secret")

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(DEFAULT_WAKE_PATH, data=raw, headers=headers)
        assert response.status == 503
        assert await response.json() == {"error": "wake disabled"}

    await _settle()
    assert recorder.events == []


# ---------------------------------------------------------------------------
# 5. body validation
# ---------------------------------------------------------------------------


async def test_missing_issue_id_or_text_is_a_400():
    adapter = _make_adapter()
    recorder = _install_recorder(adapter)
    bodies = [
        _body({"text": "check the deploy"}),
        _body({"issue_id": ISSUE_ID}),
    ]

    async with TestClient(TestServer(adapter._build_app())) as client:
        for raw in bodies:
            response = await client.post(DEFAULT_WAKE_PATH, data=raw, headers=_signed(raw))
            assert response.status == 400

    await _settle()
    assert recorder.events == []


# ---------------------------------------------------------------------------
# 6. route isolation
# ---------------------------------------------------------------------------


async def test_wake_is_not_reachable_under_the_webhook_path():
    adapter = _make_adapter()
    recorder = _install_recorder(adapter)
    raw = _body({"issue_id": ISSUE_ID, "text": "check the deploy"})
    paths = [
        f"{adapter._webhook_path}/internal/wake",
        "/p/other/internal/wake",
    ]

    async with TestClient(TestServer(adapter._build_app())) as client:
        for path in paths:
            response = await client.post(path, data=raw, headers=_signed(raw))
            assert response.status in (404, 405), path

    await _settle()
    assert recorder.events == []
