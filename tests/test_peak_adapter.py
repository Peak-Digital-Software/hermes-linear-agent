"""Regression tests for the peak/patches hardening pass on the Linear adapter.

Covers: per-issue session keying (one Hermes session per Linear ISSUE, not per
agentSession), the stop-signal key contract, strict ``mutation_policy``
parsing, the fail-closed configured-workspace check, ``/p/<profile>/`` path
validation, and the loopback default bind host.
"""

import asyncio
import hashlib
import hmac
import json
import time

import pytest
from gateway.config import PlatformConfig

from hermes_linear_agent.adapter import (
    DEFAULT_HOST,
    LinearAgentAdapter,
    LinearAgentConfigError,
)
from hermes_linear_agent.webhook import extract_context

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeLinearClient:
    def __init__(self):
        self.calls = []
        # Full create_response payloads (P16 elicitation signal assertions)
        self.activities = []
        # issue id -> agent session nodes (P12 handoff count / P13 recovery)
        self.agent_sessions = {}
        # agent session id -> comment author id (P12 sender fallback)
        self.comment_users = {}
        # issue id -> creator id (P14 sender fallback)
        self.issue_creators = {}
        self.agent_sessions_error = None
        self.issue_creator_error = None

    async def list_issue_agent_sessions(self, issue_id, limit=50):
        if self.agent_sessions_error:
            raise self.agent_sessions_error
        return list(self.agent_sessions.get(issue_id, []))

    async def get_agent_session_comment_user_id(self, agent_session_id):
        if self.agent_sessions_error:
            raise self.agent_sessions_error
        return self.comment_users.get(agent_session_id, "")

    async def get_issue_creator_user_id(self, issue_id):
        if self.issue_creator_error:
            raise self.issue_creator_error
        return self.issue_creators.get(issue_id, "")

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        self.calls.append(("thought", agent_session_id, body))
        return {"id": "thought-1"}

    async def create_response(
        self,
        agent_session_id,
        body,
        *,
        response_type="response",
        ephemeral=False,
        signal=None,
        signal_metadata=None,
    ):
        self.calls.append((response_type, agent_session_id, body))
        self.activities.append(
            {
                "agent_session_id": agent_session_id,
                "body": body,
                "response_type": response_type,
                "ephemeral": ephemeral,
                "signal": signal,
                "signal_metadata": signal_metadata,
            }
        )
        return {"id": "response-1"}

    async def create_error(self, agent_session_id, body):
        self.calls.append(("error", agent_session_id, body))
        return {"id": "error-1"}

    async def create_comment(self, issue_id, body, *, parent_id=None, mutation_policy=None):
        self.calls.append(("comment", issue_id, body, mutation_policy))
        return {"id": "comment-1"}


def _body(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _headers(raw_body, secret="secret", delivery_id="evt-1"):
    return {
        "Linear-Signature": "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest(),
        "Linear-Timestamp": str(int(time.time() * 1000)),
        "Linear-Delivery-Id": delivery_id,
    }


def _prompted_payload(event_id="evt-1", *, session_id="session-1", issue_id="issue-1", body="Keep going."):
    return {
        "action": "prompted",
        "id": event_id,
        "agentSession": {
            "id": session_id,
            "workspaceId": "workspace-1",
            "issue": {
                "id": issue_id,
                "identifier": "LIN-123",
                "title": "Fix the flaky job",
                "team": {"id": "team-1"},
            },
        },
        "agentActivity": {"id": "activity-1", "body": body, "actor": {"id": "user-1", "name": "Ada"}},
    }


def _created_payload(event_id="evt-1", *, session_id="session-1", issue_id="issue-1"):
    payload = _prompted_payload(event_id, session_id=session_id, issue_id=issue_id)
    payload["action"] = "created"
    payload["promptContext"] = "The issue has a failing CI check."
    return payload


def _make_adapter(*, extra=None, client=None):
    merged = {
        "access_token": "linear-token",
        "webhook_secret": "secret",
        "allow_all_users": True,
    }
    if extra:
        merged.update(extra)
    return LinearAgentAdapter(PlatformConfig(enabled=True, extra=merged), client=client or _FakeLinearClient())


async def _post(adapter, payload, *, profile=None, delivery_id=None):
    raw = _body(payload)
    headers = _headers(raw, delivery_id=delivery_id or payload["id"])
    return await adapter.handle_webhook(headers, raw, profile=profile)


async def _settle():
    """Let handle_message's background dispatch task reach the handler."""
    for _ in range(10):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# P3 — one Hermes session per ISSUE
# ---------------------------------------------------------------------------


def test_two_agent_sessions_on_one_issue_share_a_session_key():
    """Linear mints a new agentSession per mention; the Hermes session must not."""
    adapter = _make_adapter()
    first = _prompted_payload(session_id="session-1")
    second = _prompted_payload(session_id="session-2")

    key_first = adapter._session_key_for(extract_context(first, {}))
    key_second = adapter._session_key_for(extract_context(second, {}))

    assert key_first == key_second
    assert "issue-1" in key_first


def test_different_issues_get_different_session_keys():
    adapter = _make_adapter()
    one = _prompted_payload(issue_id="issue-1")
    two = _prompted_payload(issue_id="issue-2")

    assert adapter._session_key_for(extract_context(one, {})) != adapter._session_key_for(extract_context(two, {}))


def _gateway_key(source, *, group_sessions_per_user=True, thread_sessions_per_user=False):
    """The key the gateway runner itself derives for a source.

    ``GatewayRunner._session_key_for_source`` → ``SessionStore._generate_session_key``
    → ``gateway.session.build_session_key``. Keying history in the store happens
    THERE, so the adapter's own key is only correct if it matches this exactly.
    """
    from gateway.session import build_session_key

    return build_session_key(
        source,
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=thread_sessions_per_user,
    )


def test_session_key_matches_the_gateway_runner_key_for_the_built_source():
    adapter = _make_adapter()
    for session_id in ("session-1", "session-2"):
        context = extract_context(_prompted_payload(session_id=session_id), {})
        source = adapter._build_source(context)
        assert adapter._session_key_for(context) == _gateway_key(source)
        assert source.chat_id == "issue-1"


def test_issue_session_ignores_the_agent_session_and_the_user():
    """One Hermes session per ISSUE: a follow-up mention from another allowed
    user on the same issue lands in the same session (thread semantics), unless
    the operator opts into per-user thread sessions."""
    adapter = _make_adapter()
    first = _prompted_payload(session_id="session-1")
    second = _prompted_payload(session_id="session-2", event_id="evt-2")
    second["agentActivity"]["actor"] = {"id": "user-2", "name": "Tom"}

    assert adapter._session_key_for(extract_context(first, {})) == adapter._session_key_for(extract_context(second, {}))

    per_user = _make_adapter(extra={"thread_sessions_per_user": True})
    assert per_user._session_key_for(extract_context(first, {})) != per_user._session_key_for(
        extract_context(second, {})
    )


@pytest.mark.asyncio
async def test_replies_for_the_issue_go_to_the_newest_agent_session():
    """Both mentions run in ONE Hermes session keyed by the issue, and the
    gateway's reply (delivered on the source chat_id = the issue id) must
    resolve to the NEWEST Linear agent session, not the first one."""
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client)
    events = []

    async def capture(event):
        events.append(event)

    adapter._message_handler = capture

    await _post(adapter, _prompted_payload(session_id="session-1", event_id="evt-1"), delivery_id="evt-1")
    await _settle()
    # The newer Linear agent session on the SAME issue (the follow-up turn).
    await _post(adapter, _prompted_payload(session_id="session-2", event_id="evt-2"), delivery_id="evt-2")
    await _settle()

    # One session, keyed by the issue — not one per Linear agent session.
    assert [event.source.chat_id for event in events] == ["issue-1", "issue-1"]
    assert events[0].source.thread_id == events[1].source.thread_id == "issue-1"
    assert adapter._session_key_for(
        extract_context(_prompted_payload(session_id="session-1"), {})
    ) == adapter._session_key_for(extract_context(_prompted_payload(session_id="session-2"), {}))

    # The gateway replies to the inbound event's own source chat_id (issue id).
    newer = events[-1]
    result = await adapter.send(newer.source.chat_id, "Done — here is what I found.")
    assert result.success is True
    responses = [call for call in client.calls if call[0] == "response"]
    assert responses[-1][1] == "session-2"


@pytest.mark.asyncio
async def test_unknown_chat_id_falls_back_to_an_issue_comment():
    """After a restart (or for cron/home delivery) the per-issue session map is
    empty: the reply must still land, as a comment on the issue."""
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client, extra={"mutation_policy": {"create_comments": True}})

    result = await adapter.send("issue-1", "Here is the scheduled result.")

    assert result.success is True
    assert [call[0] for call in client.calls] == ["comment"]
    assert client.calls[0][1] == "issue-1"


@pytest.mark.asyncio
async def test_stop_signal_from_a_new_agent_session_interrupts_the_issue_turn():
    """A follow-up session's `stop` must reach the turn started by an earlier
    agent session on the same issue — that only works when both key per-issue."""
    adapter = _make_adapter()
    interrupts = []

    async def fake_interrupt(session_key, chat_id):
        interrupts.append((session_key, chat_id))

    adapter.interrupt_session_activity = fake_interrupt

    issue_key = adapter._session_key_for(extract_context(_prompted_payload(session_id="session-1"), {}))
    adapter._active_sessions[issue_key] = object()  # the turn session-1 started

    payload = _prompted_payload(session_id="session-2", event_id="evt-stop")
    payload["agentActivity"]["signal"] = "stop"
    response, status = await _post(adapter, payload, delivery_id="evt-stop")

    assert status == 200 and response["status"] == "stopped"
    # Same per-issue key dispatch used, and the chat_id is the issue id.
    assert interrupts == [(issue_key, "issue-1")]


# ---------------------------------------------------------------------------
# P4 — the stop-signal key is exactly the dispatch key (named profile)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_key_for_matches_dispatch_key_with_a_named_profile():
    adapter = _make_adapter()
    adapter.set_owner_profile("work")
    captured = []

    async def capture_background(event, session_key):
        captured.append(session_key)

    async def message_handler(event):
        return None

    adapter._message_handler = message_handler
    adapter._process_message_background = capture_background

    payload = _prompted_payload(event_id="evt-profile")
    _, status = await _post(adapter, payload, profile="work", delivery_id="evt-profile")
    await _settle()

    assert status == 200
    assert captured, "real handle_message never reached background dispatch"

    key = adapter._session_key_for(extract_context(payload, {}), profile="work")
    assert key == captured[0]
    # Profile namespacing applied the same way on both paths.
    assert key.startswith("agent:work:")


# ---------------------------------------------------------------------------
# P5 — strict mutation_policy parsing
# ---------------------------------------------------------------------------


def test_mutation_policy_string_false_disables_the_write():
    adapter = _make_adapter(
        extra={"mutation_policy": {"create_comments": "false", "update_issues": "TRUE", "delete_customers": "No"}}
    )

    assert adapter._mutation_policy["create_comments"] is False
    assert adapter._mutation_policy["update_issues"] is True
    assert adapter._mutation_policy["delete_customers"] is False


def test_mutation_policy_numeric_strings_convert():
    adapter = _make_adapter(extra={"mutation_policy": {"create_issues": "1", "create_labels": "0"}})
    assert adapter._mutation_policy["create_issues"] is True
    assert adapter._mutation_policy["create_labels"] is False


def test_mutation_policy_garbage_value_raises_config_error():
    with pytest.raises(LinearAgentConfigError) as excinfo:
        _make_adapter(extra={"mutation_policy": {"update_issues": "maybe"}})
    assert "update_issues" in str(excinfo.value)


def test_mutation_policy_non_string_non_bool_raises_config_error():
    with pytest.raises(LinearAgentConfigError):
        _make_adapter(extra={"mutation_policy": {"update_issues": 2}})


def test_mutation_policy_unknown_keys_are_not_copied_into_the_policy():
    adapter = _make_adapter(extra={"mutation_policy": {"create_comment": True, "update_issues": True}})
    assert "create_comment" not in adapter._mutation_policy
    assert adapter._mutation_policy["update_issues"] is True


# ---------------------------------------------------------------------------
# P6 — configured workspace: missing id fails closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configured_workspace_rejects_missing_workspace_id():
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client, extra={"workspace_id": "workspace-1"})
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture

    payload = _created_payload()
    del payload["agentSession"]["workspaceId"]
    response, status = await _post(adapter, payload, delivery_id="evt-ws-missing")

    assert status == 200
    assert response["status"] == "ignored"
    assert response["reason"] == "missing workspace"
    assert dispatched == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_configured_workspace_still_accepts_its_own_workspace():
    adapter = _make_adapter(extra={"workspace_id": "workspace-1"})
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture

    response, status = await _post(adapter, _created_payload(), delivery_id="evt-ws-ok")

    assert status == 200
    assert response["status"] == "accepted"
    assert len(dispatched) == 1


# ---------------------------------------------------------------------------
# P7 — /p/<profile>/ path validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_path_mismatch_is_rejected_before_processing():
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client)
    adapter.set_owner_profile("alpha")
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture

    response, status = await _post(adapter, _created_payload(), profile="beta", delivery_id="evt-beta")

    assert status == 404
    assert response["error"] == "Unknown profile"
    assert dispatched == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_profile_path_matching_the_own_profile_is_accepted():
    adapter = _make_adapter()
    adapter.set_owner_profile("alpha")
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture

    _, status = await _post(adapter, _created_payload(), profile="alpha", delivery_id="evt-alpha")

    assert status == 200
    assert len(dispatched) == 1


@pytest.mark.asyncio
async def test_profile_path_fails_closed_when_it_names_another_profile(monkeypatch):
    """No declared owner profile: only a prefix naming THIS process's profile
    may pass (single-profile gateway); everything else is a 404."""
    import hermes_cli.profiles as profiles_mod

    adapter = _make_adapter()
    assert adapter._own_profile_name() == ""
    monkeypatch.setattr(profiles_mod, "profile_matches_home", lambda name, home=None: False)

    response, status = await _post(adapter, _created_payload(), profile="someone-else", delivery_id="evt-x")

    assert status == 404
    assert response["error"] == "Unknown profile"


@pytest.mark.asyncio
async def test_unprefixed_webhook_is_unaffected(monkeypatch):
    import hermes_cli.profiles as profiles_mod

    adapter = _make_adapter()
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    monkeypatch.setattr(profiles_mod, "profile_matches_home", lambda name, home=None: False)

    _, status = await _post(adapter, _created_payload(), delivery_id="evt-bare")

    assert status == 200
    assert len(dispatched) == 1


# ---------------------------------------------------------------------------
# P8 — loopback default bind
# ---------------------------------------------------------------------------


def test_default_bind_host_is_loopback():
    adapter = _make_adapter()
    assert DEFAULT_HOST == "127.0.0.1"
    assert adapter._host == "127.0.0.1"


def test_bind_host_is_still_overridable():
    adapter = _make_adapter(extra={"webhook_host": "0.0.0.0"})
    assert adapter._host == "0.0.0.0"


def test_str_env_reads_the_profile_secret_scope(monkeypatch):
    """P9: under multiplexing a profile's .env lives only in its secret scope."""
    import gateway.platforms._shared as shared

    from hermes_linear_agent import adapter as adapter_mod

    monkeypatch.delenv("LINEAR_AGENT_CLIENT_ID", raising=False)
    monkeypatch.setattr(
        shared,
        "get_scoped_secret",
        lambda name, default=None, **_: {"LINEAR_AGENT_CLIENT_ID": " scoped-id "}.get(name, default),
    )
    assert adapter_mod._str_env("LINEAR_AGENT_CLIENT_ID") == "scoped-id"
    assert adapter_mod._str_env("LINEAR_AGENT_MISSING") == ""


# ---------------------------------------------------------------------------
# P12 — peer agent handoffs with a Linear-derived loop cap
# ---------------------------------------------------------------------------


def _handoff_created_payload(
    event_id="evt-peer",
    *,
    session_id="session-peer",
    issue_id="issue-1",
    comment_user_id="peer-1",
    include_comment=True,
):
    """A `created` session another agent app opened (creator is null)."""
    payload = {
        "action": "created",
        "id": event_id,
        "promptContext": "Handoff from the triage agent.",
        "agentSession": {
            "id": session_id,
            "workspaceId": "workspace-1",
            "status": "pending",
            "commentId": "comment-1",
            "issue": {
                "id": issue_id,
                "identifier": "LIN-123",
                "title": "Fix the flaky job",
                "team": {"id": "team-1"},
            },
        },
    }
    if include_comment:
        payload["agentSession"]["comment"] = {
            "id": "comment-1",
            "body": "<<mention>>",
            "userId": comment_user_id,
        }
    return payload


def _app_session(session_id, *, created_at, app_user_id="peer-1"):
    """An app-initiated agent session: no human creator, app comment author."""
    return {
        "id": session_id,
        "createdAt": created_at,
        "creator": None,
        "appUser": {"id": app_user_id, "name": "Agent"},
        "comment": {"id": f"c-{session_id}", "user": {"id": app_user_id, "name": "Agent", "app": True}},
    }


def _human_session(session_id, *, created_at, user_id="user-1"):
    return {
        "id": session_id,
        "createdAt": created_at,
        "creator": {"id": user_id, "name": "Ada", "app": False},
        "appUser": {"id": "app-1", "name": "Hermes"},
        "comment": {"id": f"c-{session_id}", "user": {"id": user_id, "name": "Ada", "app": False}},
    }


async def _capture_events(adapter):
    events = []

    async def capture(event):
        events.append(event)

    adapter._message_handler = capture
    return events


@pytest.mark.asyncio
async def test_creatorless_session_takes_its_sender_from_the_comment_user():
    """A peer session has no agentActivity.userId and no creator: the sender
    is the session comment's author (payload field agentSession.comment.userId)."""
    client = _FakeLinearClient()
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "allow_all_users": False, "allowed_users": ["peer-1"]},
    )
    events = await _capture_events(adapter)

    context = extract_context(_handoff_created_payload(), {})
    assert context.actor_user_id == ""
    assert context.comment_user_id == "peer-1"

    _, status = await _post(adapter, _handoff_created_payload(), delivery_id="evt-peer")
    await _settle()

    assert status == 200
    assert len(events) == 1
    assert events[0].source.user_id == "peer-1"


@pytest.mark.asyncio
async def test_sender_falls_back_to_a_graphql_comment_lookup_when_the_payload_omits_it():
    client = _FakeLinearClient()
    client.comment_users["session-peer"] = "peer-1"
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "allow_all_users": False, "allowed_users": ["peer-1"]},
    )
    events = await _capture_events(adapter)

    payload = _handoff_created_payload(include_comment=False)
    _, status = await _post(adapter, payload, delivery_id="evt-peer")
    await _settle()

    assert status == 200
    assert [event.source.user_id for event in events] == ["peer-1"]


@pytest.mark.asyncio
async def test_peer_is_allowed_and_non_peer_and_self_are_refused():
    peer = _make_adapter(
        extra={
            "app_user_id": "app-1",
            "allow_all_users": False,
            "peer_app_user_ids": ["peer-1"],
        }
    )
    events = await _capture_events(peer)
    _, status = await _post(peer, _handoff_created_payload(), delivery_id="evt-peer")
    await _settle()
    assert status == 200
    assert len(events) == 1

    stranger = _make_adapter(
        extra={"app_user_id": "app-1", "allow_all_users": False, "peer_app_user_ids": ["peer-1"]}
    )
    stranger_events = await _capture_events(stranger)
    _, status = await _post(
        stranger,
        _handoff_created_payload(comment_user_id="someone-else"),
        delivery_id="evt-stranger",
    )
    await _settle()
    assert status == 403
    assert stranger_events == []

    # Our own app user id is never a valid sender, even when it is listed and
    # every user is allowed.
    own = _make_adapter(
        extra={
            "app_user_id": "app-1",
            "allow_all_users": True,
            "peer_app_user_ids": ["app-1"],
        }
    )
    own_events = await _capture_events(own)
    _, status = await _post(own, _handoff_created_payload(comment_user_id="app-1"), delivery_id="evt-self")
    await _settle()
    assert status == 403
    assert own_events == []


@pytest.mark.asyncio
async def test_handoff_cap_allows_two_and_refuses_three():
    client = _FakeLinearClient()
    client.agent_sessions["issue-1"] = [
        _human_session("session-human", created_at="2026-01-01T00:00:00.000Z"),
        _app_session("session-a", created_at="2026-01-01T00:01:00.000Z"),
        _app_session("session-b", created_at="2026-01-01T00:02:00.000Z"),
    ]
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1", "peer_app_user_ids": ["peer-1"]})
    events = await _capture_events(adapter)

    _, status = await _post(adapter, _handoff_created_payload(), delivery_id="evt-peer")
    await _settle()
    assert status == 200
    assert len(events) == 1, "a peer session at the cap must still be dispatched"

    # A third handoff on the issue saturates it (max defaults to 2).
    client.agent_sessions["issue-1"].append(_app_session("session-c", created_at="2026-01-01T00:03:00.000Z"))
    client.calls.clear()
    response, status = await _post(
        adapter, _handoff_created_payload(session_id="session-d", event_id="evt-4"), delivery_id="evt-4"
    )
    await _settle()

    assert response["status"] == "refused"
    assert response["handoffs"] == 3
    assert status == 200
    assert len(events) == 1, "a refused handoff must not reach Hermes"
    responses = [call for call in client.calls if call[0] == "response"]
    assert len(responses) == 1
    assert responses[0][1] == "session-d"
    assert "already has 3 agent-to-agent handoffs" in responses[0][2]
    assert "Tom needs to step in" in responses[0][2]
    # No acknowledgement thought either: the refusal is the only activity.
    assert [call[0] for call in client.calls] == ["response"]


@pytest.mark.asyncio
async def test_handoff_cap_counts_creator_app_sessions_too():
    client = _FakeLinearClient()
    client.agent_sessions["issue-1"] = [
        {
            "id": f"session-{i}",
            "createdAt": f"2026-01-01T00:0{i}:00.000Z",
            "creator": {"id": "peer-1", "name": "Agent", "app": True},
            "appUser": {"id": "peer-1"},
            "comment": None,
        }
        for i in range(3)
    ]
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1", "peer_app_user_ids": ["peer-1"]})

    response, status = await _post(adapter, _handoff_created_payload(), delivery_id="evt-peer")
    assert status == 200 and response["status"] == "refused"


@pytest.mark.asyncio
async def test_handoff_cap_fails_closed_when_the_count_query_errors():
    client = _FakeLinearClient()
    client.agent_sessions_error = RuntimeError("linear is down")
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1", "peer_app_user_ids": ["peer-1"]})
    events = await _capture_events(adapter)

    response, status = await _post(adapter, _handoff_created_payload(), delivery_id="evt-peer")
    await _settle()

    assert status == 200
    assert response["status"] == "refused"
    assert events == []
    assert [call[0] for call in client.calls] == ["error"]
    assert client.calls[0][1] == "session-peer"


@pytest.mark.asyncio
async def test_human_sessions_never_hit_the_handoff_cap():
    """Only peer senders are capped; a human mention dispatches regardless."""
    client = _FakeLinearClient()
    client.agent_sessions["issue-1"] = [_app_session(f"session-{i}", created_at=f"2026-01-01T00:0{i}:00.000Z") for i in range(5)]
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1"})
    events = await _capture_events(adapter)

    _, status = await _post(adapter, _prompted_payload(), delivery_id="evt-human")
    await _settle()

    assert status == 200
    assert len(events) == 1
    assert client.calls == []


# ---------------------------------------------------------------------------
# P13 — recover the reply session from Linear after a restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_recovers_the_newest_own_agent_session_after_a_restart():
    client = _FakeLinearClient()
    client.agent_sessions["issue-1"] = [
        {"id": "session-own-old", "createdAt": "2026-01-01T00:00:00.000Z", "appUser": {"id": "app-1"}},
        {"id": "session-other", "createdAt": "2026-01-01T00:05:00.000Z", "appUser": {"id": "app-other"}},
        {"id": "session-own-new", "createdAt": "2026-01-01T00:02:00.000Z", "appUser": {"id": "app-1"}},
    ]
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1"})
    assert adapter._latest_agent_session == {}

    result = await adapter.send("issue-1", "Here is the result.")

    assert result.success is True
    assert [call[0] for call in client.calls] == ["response"]
    assert client.calls[0][1] == "session-own-new"
    assert adapter._latest_agent_session["issue-1"] == "session-own-new"


@pytest.mark.asyncio
async def test_send_falls_back_to_a_comment_when_the_recovery_lookup_fails():
    client = _FakeLinearClient()
    client.agent_sessions_error = RuntimeError("linear is down")
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1", "mutation_policy": {"create_comments": True}})

    result = await adapter.send("issue-1", "Here is the result.")

    assert result.success is True
    assert [call[0] for call in client.calls] == ["comment"]
    assert client.calls[0][1] == "issue-1"


@pytest.mark.asyncio
async def test_send_does_not_recover_for_a_synthetic_issue_update_target():
    client = _FakeLinearClient()
    client.agent_sessions["issue-1"] = [
        {"id": "session-own", "createdAt": "2026-01-01T00:00:00.000Z", "appUser": {"id": "app-1"}},
    ]
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1", "mutation_policy": {"create_comments": True}})

    result = await adapter.send("update:issue-1", "Here is the result.")

    assert result.success is True
    assert [call[0] for call in client.calls] == ["comment"]


# ---------------------------------------------------------------------------
# P14 — a session started by a Linear automation rule resolves to the issue's
# creator (Linear authors the opening prompt as our own app user, so the
# agentActivity.userId is the machine prompt, not a sender)
# ---------------------------------------------------------------------------


def _rule_delegated_payload(
    event_id="evt-rule",
    *,
    session_id="session-rule",
    issue_id="issue-1",
    activity_user_id="app-1",
    creator_id=None,
    comment_user_id=None,
):
    """A `created` session an automation rule delegated to this app.

    Matches the live shape: ``agentSession.creator`` is null, no session
    comment, and the only activity is the machine prompt authored by this app.
    """
    payload = {
        "action": "created",
        "id": event_id,
        "promptContext": "Work on LIN-123",
        "agentSession": {
            "id": session_id,
            "workspaceId": "workspace-1",
            "status": "pending",
            "creator": None,
            "issue": {
                "id": issue_id,
                "identifier": "LIN-123",
                "title": "Fix the flaky job",
                "team": {"id": "team-1"},
            },
        },
        "agentActivity": {
            "id": "activity-1",
            "userId": activity_user_id,
            "content": {"type": "prompt", "body": "Work on LIN-123"},
            "user": {"id": activity_user_id, "name": "Hermes"},
        },
    }
    if creator_id:
        payload["agentSession"]["creatorId"] = creator_id
    if comment_user_id:
        payload["agentSession"]["comment"] = {"id": "comment-1", "userId": comment_user_id}
    return payload


def _rule_adapter(client, **extra):
    base = {"app_user_id": "app-1", "allow_all_users": False}
    base.update(extra)
    return _make_adapter(client=client, extra=base)


def test_extract_context_keeps_the_activity_author_and_creator_apart():
    context = extract_context(_rule_delegated_payload(), {})
    assert context.actor_user_id == "app-1"
    assert context.sender_source == "agentActivity.userId"
    assert context.activity_user_id == "app-1"
    assert context.session_creator_id == ""
    assert context.issue_creator_id == ""


@pytest.mark.asyncio
async def test_rule_delegated_session_takes_its_sender_from_the_issue_creator():
    """ENG-2216: the rule's opening prompt is authored by our own app; the
    sender is the human who created the issue."""
    client = _FakeLinearClient()
    client.issue_creators["issue-1"] = "tom-1"
    adapter = _rule_adapter(client, allowed_users=["tom-1"])
    events = await _capture_events(adapter)

    _, status = await _post(adapter, _rule_delegated_payload(), delivery_id="evt-rule")
    await _settle()

    assert status == 200
    assert [event.source.user_id for event in events] == ["tom-1"]


@pytest.mark.asyncio
async def test_rule_delegated_session_refuses_an_unknown_issue_creator():
    client = _FakeLinearClient()
    client.issue_creators["issue-1"] = "stranger-9"
    adapter = _rule_adapter(client, allowed_users=["tom-1"])
    events = await _capture_events(adapter)

    _, status = await _post(adapter, _rule_delegated_payload(), delivery_id="evt-rule")
    await _settle()

    assert status == 403
    assert events == []


@pytest.mark.asyncio
async def test_rule_delegated_session_refuses_when_the_issue_creator_is_this_app():
    client = _FakeLinearClient()
    client.issue_creators["issue-1"] = "app-1"
    # Even "allow every user" must not accept our own app as the sender.
    adapter = _rule_adapter(client, allow_all_users=True, allowed_users=["app-1"])
    events = await _capture_events(adapter)

    _, status = await _post(adapter, _rule_delegated_payload(), delivery_id="evt-rule")
    await _settle()

    assert status == 403
    assert events == []


@pytest.mark.asyncio
async def test_rule_delegated_peer_issue_creator_goes_through_the_handoff_cap():
    client = _FakeLinearClient()
    client.issue_creators["issue-1"] = "peer-1"
    client.agent_sessions["issue-1"] = [
        _app_session(f"session-{i}", created_at=f"2026-01-01T00:0{i}:00.000Z") for i in range(3)
    ]
    adapter = _rule_adapter(client, peer_app_user_ids=["peer-1"])
    events = await _capture_events(adapter)

    response, status = await _post(adapter, _rule_delegated_payload(), delivery_id="evt-rule")
    await _settle()

    assert status == 200
    assert response["status"] == "refused"
    assert events == []
    assert [call[0] for call in client.calls] == ["response"]
    assert "already has 3 agent-to-agent handoffs" in client.calls[0][2]


@pytest.mark.asyncio
async def test_rule_delegated_session_refuses_when_the_issue_creator_lookup_fails():
    client = _FakeLinearClient()
    client.issue_creator_error = RuntimeError("linear is down")
    adapter = _rule_adapter(client, allowed_users=["tom-1"])
    events = await _capture_events(adapter)

    _, status = await _post(adapter, _rule_delegated_payload(), delivery_id="evt-rule")
    await _settle()

    assert status == 403
    assert events == []


@pytest.mark.asyncio
async def test_prompted_self_echo_is_still_refused_even_with_an_allowed_issue_creator():
    """P14 only reinterprets the activity author on `created`; our own activity
    echoing back as a prompted event stays a rejected sender."""
    client = _FakeLinearClient()
    client.issue_creators["issue-1"] = "tom-1"
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "allow_all_users": True, "allowed_users": ["tom-1"]},
    )
    events = await _capture_events(adapter)

    payload = _prompted_payload()
    payload["agentActivity"]["actor"] = {"id": "app-1", "name": "Hermes"}
    _, status = await _post(adapter, payload, delivery_id="evt-echo")
    await _settle()

    assert status == 403
    assert events == []


# ---------------------------------------------------------------------------
# P16 — clarify: `select` elicitations carry signalMetadata.options, and the
# elicitation recovers the issue's session after a restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clarify_with_choices_sends_select_signal_with_one_option_per_choice():
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1"})
    adapter._latest_agent_session["issue-1"] = "session-1"

    result = await adapter.send_clarify(
        "issue-1",
        "Which package should change?",
        ["frontend", "backend"],
        "clarify-1",
        "session-key",
    )

    assert result.success is True
    assert len(client.activities) == 1
    activity = client.activities[0]
    assert activity["agent_session_id"] == "session-1"
    assert activity["response_type"] == "elicitation"
    assert activity["signal"] == "select"
    assert activity["signal_metadata"] == {
        "options": [
            {"label": "frontend", "value": "frontend"},
            {"label": "backend", "value": "backend"},
        ]
    }
    # The numbered list stays in the body as the free-text fallback.
    assert "1. frontend" in activity["body"]
    assert "2. backend" in activity["body"]



@pytest.mark.asyncio
async def test_clarify_without_choices_sends_a_plain_elicitation():
    client = _FakeLinearClient()
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1"})
    adapter._latest_agent_session["issue-1"] = "session-1"

    result = await adapter.send_clarify(
        "issue-1",
        "What should the timeout be?",
        None,
        "clarify-1",
        "session-key",
    )

    assert result.success is True
    assert len(client.activities) == 1
    activity = client.activities[0]
    assert activity["response_type"] == "elicitation"
    assert activity["signal"] is None
    assert activity["signal_metadata"] is None


@pytest.mark.asyncio
async def test_clarify_recovers_the_newest_own_agent_session_after_a_restart():
    client = _FakeLinearClient()
    client.agent_sessions["issue-1"] = [
        {"id": "session-own-old", "createdAt": "2026-01-01T00:00:00.000Z", "appUser": {"id": "app-1"}},
        {"id": "session-other", "createdAt": "2026-01-01T00:05:00.000Z", "appUser": {"id": "app-other"}},
        {"id": "session-own-new", "createdAt": "2026-01-01T00:02:00.000Z", "appUser": {"id": "app-1"}},
    ]
    adapter = _make_adapter(client=client, extra={"app_user_id": "app-1"})
    assert adapter._latest_agent_session == {}

    result = await adapter.send_clarify(
        "issue-1",
        "Which package should change?",
        ["frontend", "backend"],
        "clarify-1",
        "session-key",
    )

    assert result.success is True
    # The elicitation goes into the recovered session, not an issue comment.
    assert [call[0] for call in client.calls] == ["elicitation"]
    assert len(client.activities) == 1
    assert client.activities[0]["agent_session_id"] == "session-own-new"
    assert client.activities[0]["signal"] == "select"


@pytest.mark.asyncio
async def test_clarify_falls_back_to_the_base_text_path_when_there_is_no_session_at_all():
    client = _FakeLinearClient()
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "mutation_policy": {"create_comments": True}},
    )
    assert adapter._latest_agent_session == {}

    result = await adapter.send_clarify(
        "issue-1",
        "Which package should change?",
        ["frontend", "backend"],
        "clarify-1",
        "session-key",
    )

    assert result.success is True
    assert client.activities == []
    assert [call[0] for call in client.calls] == ["comment"]
    assert client.calls[0][1] == "issue-1"
