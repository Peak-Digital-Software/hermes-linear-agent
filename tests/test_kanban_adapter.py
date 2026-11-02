"""Adapter wiring for the kanban ⇄ Linear sync (ENG-2274 W4).

``/hermes/linear-issues`` is a data-webhook entry point separate from the
agent-session webhook: its own path, its own secret, no sender allowlist, no
dedup, no auto-start. The sync itself (``kanban_sync.KanbanSync``) is faked
here — these tests cover only the wiring in ``adapter.py``.
"""

import asyncio
import hashlib
import hmac
import json
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer
from gateway.config import PlatformConfig

from hermes_linear_agent import adapter as adapter_mod
from hermes_linear_agent.adapter import (
    ISSUE_WEBHOOK_PATH,
    LinearAgentAdapter,
    _apply_yaml_config,
)

AGENT_SECRET = "agent-secret"
ISSUE_SECRET = "issue-secret"


# ---------------------------------------------------------------------------
# fakes / helpers
# ---------------------------------------------------------------------------


class _FakeLinearClient:
    """Just enough client for adapter construction; nothing is sent."""

    configured = True

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        return {"id": "thought-1"}

    async def create_response(self, agent_session_id, body, **kwargs):
        return {"id": "response-1"}

    async def create_error(self, agent_session_id, body):
        return {"id": "error-1"}


class _FakeKanbanSync:
    """Records construction, payloads and the tasks ``start`` handed out."""

    instances: list["_FakeKanbanSync"] = []

    def __init__(self, client, team_key="ENG", *, wake=None):
        self.client = client
        self.team_key = team_key
        self.wake = wake
        self.payloads: list[dict] = []
        self.tasks: list[asyncio.Task] = []
        self.started = False
        type(self).instances.append(self)

    def start(self):
        self.started = True
        self.tasks = [asyncio.create_task(asyncio.sleep(3600)) for _ in range(2)]
        return list(self.tasks)

    async def handle_issue_webhook(self, payload):
        self.payloads.append(payload)
        return f"handled {payload.get('action')}"

    async def reconcile(self):
        return 0


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeKanbanSync.instances = []
    yield
    _FakeKanbanSync.instances = []


def _use_fake_kanban(monkeypatch):
    monkeypatch.setattr(adapter_mod, "_kanban_sync_class", lambda: _FakeKanbanSync)


def _make_adapter(*, extra=None, client=None):
    merged = {
        "access_token": "linear-token",
        "webhook_secret": AGENT_SECRET,
        "issue_webhook_secret": ISSUE_SECRET,
        "allow_all_users": True,
        "app_user_id": "app-1",
        "workspace_id": "workspace-1",
    }
    if extra:
        merged.update(extra)
    return LinearAgentAdapter(
        PlatformConfig(enabled=True, extra=merged),
        client=client or _FakeLinearClient(),
    )


def _issue_payload(*, action="update", timestamp=None):
    payload = {
        "action": action,
        "type": "Issue",
        "data": {
            "id": "issue-1",
            "identifier": "ENG-1",
            "teamId": "team-1",
            "stateId": "state-done",
        },
        "updatedFrom": {"stateId": "state-working"},
        "webhookTimestamp": int((timestamp if timestamp is not None else time.time()) * 1000),
    }
    return payload


def _body(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _signed(raw, *, secret=ISSUE_SECRET, timestamp=None):
    stamp = str(int(time.time()) if timestamp is None else int(timestamp))
    signature = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return {"Linear-Signature": signature, "Linear-Timestamp": stamp}


def _signed_body_only(raw, *, secret=ISSUE_SECRET):
    """Signature header without a timestamp header (body-anchored freshness)."""
    signature = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return {"Linear-Signature": signature}


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


def _lock_free(monkeypatch):
    import gateway.status as status_mod
    import gateway.status_scoped_locks as scoped_locks_mod

    monkeypatch.setattr(
        status_mod,
        "acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        scoped_locks_mod,
        "release_scoped_lock",
        lambda scope, identity: None,
    )


# ---------------------------------------------------------------------------
# 1. config plumbing
# ---------------------------------------------------------------------------


def test_config_allowlist_carries_kanban_options():
    seeded = _apply_yaml_config(
        {},
        {"kanban_sync": True, "issue_webhook_secret": ISSUE_SECRET},
    )
    assert seeded == {"kanban_sync": True, "issue_webhook_secret": ISSUE_SECRET}


def test_flag_defaults_off():
    adapter = _make_adapter()
    assert adapter._kanban_sync is False
    assert adapter._kanban is None


def test_flag_and_secret_read_from_scoped_env(monkeypatch):
    monkeypatch.setenv("LINEAR_AGENT_KANBAN_SYNC", "true")
    monkeypatch.setenv("LINEAR_ISSUE_WEBHOOK_SECRET", ISSUE_SECRET)
    adapter = LinearAgentAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "access_token": "linear-token",
                "webhook_secret": AGENT_SECRET,
                "allow_all_users": True,
            },
        ),
        client=_FakeLinearClient(),
    )
    assert adapter._kanban_sync is True
    assert adapter._issue_webhook_secret == ISSUE_SECRET


def test_explicit_yaml_false_beats_env(monkeypatch):
    monkeypatch.setenv("LINEAR_AGENT_KANBAN_SYNC", "true")
    adapter = _make_adapter(extra={"kanban_sync": False})
    assert adapter._kanban_sync is False


# ---------------------------------------------------------------------------
# 2. the /hermes/linear-issues route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_webhook_accepts_signed_payload(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True})
    adapter._kanban = _FakeKanbanSync(adapter._client)
    payload = _issue_payload()
    raw = _body(payload)

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(ISSUE_WEBHOOK_PATH, data=raw, headers=_signed(raw))
        assert response.status == 200

    await _settle()
    assert adapter._kanban.payloads == [payload]


@pytest.mark.asyncio
async def test_issue_webhook_rejects_wrong_secret(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True})
    adapter._kanban = _FakeKanbanSync(adapter._client)
    raw = _body(_issue_payload())

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(
            ISSUE_WEBHOOK_PATH,
            data=raw,
            headers=_signed(raw, secret=AGENT_SECRET),
        )
        assert response.status == 401

    await _settle()
    assert adapter._kanban.payloads == []


@pytest.mark.asyncio
async def test_issue_webhook_rejects_missing_signature(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True})
    adapter._kanban = _FakeKanbanSync(adapter._client)
    raw = _body(_issue_payload())

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(ISSUE_WEBHOOK_PATH, data=raw)
        assert response.status == 401

    await _settle()
    assert adapter._kanban.payloads == []


@pytest.mark.asyncio
async def test_issue_webhook_fails_closed_without_issue_secret(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(
        extra={"kanban_sync": True, "issue_webhook_secret": ""},
    )
    monkeypatch.delenv("LINEAR_ISSUE_WEBHOOK_SECRET", raising=False)
    adapter._issue_webhook_secret = ""
    adapter._kanban = _FakeKanbanSync(adapter._client)
    raw = _body(_issue_payload())

    # Even a "correctly" signed body is rejected: there is nothing to verify.
    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(ISSUE_WEBHOOK_PATH, data=raw, headers=_signed(raw, secret=""))
        assert response.status == 401

    await _settle()
    assert adapter._kanban.payloads == []


@pytest.mark.asyncio
async def test_issue_webhook_route_absent_when_flag_off(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter()
    raw = _body(_issue_payload())

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(ISSUE_WEBHOOK_PATH, data=raw, headers=_signed(raw))
        assert response.status in (404, 405)


@pytest.mark.asyncio
async def test_issue_webhook_rejects_stale_body_timestamp(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True})
    adapter._kanban = _FakeKanbanSync(adapter._client)
    raw = _body(_issue_payload(timestamp=time.time() - 3600))

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(
            ISSUE_WEBHOOK_PATH,
            data=raw,
            headers=_signed_body_only(raw),
        )
        assert response.status == 400

    await _settle()
    assert adapter._kanban.payloads == []


@pytest.mark.asyncio
async def test_issue_webhook_without_connected_sync_still_acks(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True})
    raw = _body(_issue_payload())

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(ISSUE_WEBHOOK_PATH, data=raw, headers=_signed(raw))
        assert response.status == 200

    await _settle()
    assert _FakeKanbanSync.instances == []


# ---------------------------------------------------------------------------
# 3. isolation from the agent-session lane
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_webhook_rejects_oversized_body(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True, "max_body_bytes": 64})
    adapter._kanban = _FakeKanbanSync(adapter._client)
    raw = _body(_issue_payload())
    assert len(raw) > 64

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(ISSUE_WEBHOOK_PATH, data=raw, headers=_signed(raw))
        assert response.status == 413

    await _settle()
    assert adapter._kanban.payloads == []


@pytest.mark.asyncio
async def test_agent_session_payload_on_issue_path_never_dispatches(monkeypatch):
    """An agent-session payload signed with the ISSUE secret is data only."""
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True})
    adapter._kanban = _FakeKanbanSync(adapter._client)
    dispatched = []

    async def _record(event):
        dispatched.append(event)

    adapter.handle_message = _record

    payload = {
        "action": "created",
        "type": "AgentSession",
        "agentSession": {
            "id": "session-1",
            "workspaceId": "workspace-1",
            "issue": {
                "id": "issue-1",
                "identifier": "ENG-1",
                "title": "Do the thing",
                "team": {"id": "team-1"},
            },
        },
        "agentActivity": {
            "id": "activity-1",
            "body": "Please do the thing",
            "actor": {"id": "user-1", "name": "Ada"},
        },
        "webhookTimestamp": int(time.time() * 1000),
    }
    raw = _body(payload)

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(ISSUE_WEBHOOK_PATH, data=raw, headers=_signed(raw))
        assert response.status == 200

    await _settle()
    assert dispatched == []
    assert adapter._kanban.payloads == [payload]


@pytest.mark.asyncio
async def test_agent_session_path_rejects_issue_secret(monkeypatch):
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"kanban_sync": True})
    adapter._kanban = _FakeKanbanSync(adapter._client)
    raw = _body(_issue_payload())

    async with TestClient(TestServer(adapter._build_app())) as client:
        response = await client.post(
            adapter._webhook_path,
            data=raw,
            headers=_signed(raw, secret=ISSUE_SECRET),
        )
        assert response.status == 401

    await _settle()
    assert adapter._kanban.payloads == []


# ---------------------------------------------------------------------------
# 4. lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_starts_sync_tasks_when_enabled(monkeypatch):
    _lock_free(monkeypatch)
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"port": 0, "kanban_sync": True})

    assert await adapter.connect() is True
    try:
        assert len(_FakeKanbanSync.instances) == 1
        sync = _FakeKanbanSync.instances[0]
        assert sync.client is adapter._client
        # The sync wakes sessions through the adapter's own dispatcher (bound
        # method), so issueUnblocked and child rollups reach a real turn.
        assert sync.wake == adapter._dispatch_wake
        assert sync.started is True
        assert adapter._kanban is sync
        assert len(sync.tasks) == 2
        assert all(task in adapter._background_tasks for task in sync.tasks)
    finally:
        await adapter.disconnect()

    assert all(task.cancelled() or task.done() for task in sync.tasks)


@pytest.mark.asyncio
async def test_connect_with_flag_off_starts_nothing(monkeypatch):
    _lock_free(monkeypatch)
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"port": 0})

    assert await adapter.connect() is True
    try:
        assert _FakeKanbanSync.instances == []
        assert adapter._kanban is None
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cancels_sync_tasks(monkeypatch):
    _lock_free(monkeypatch)
    _use_fake_kanban(monkeypatch)
    adapter = _make_adapter(extra={"port": 0, "kanban_sync": True})

    assert await adapter.connect() is True
    sync = _FakeKanbanSync.instances[0]
    tasks = list(sync.tasks)
    assert not any(task.done() for task in tasks)

    await adapter.disconnect()

    assert all(task.done() for task in tasks)
    assert adapter._background_tasks == set()
