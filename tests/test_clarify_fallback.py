"""P17: the core text-clarify fallback must not post a second Linear activity.

The gateway's clarify lane re-sends the prompt as plain text whenever the native send is
classified ``failed`` (``~/.hermes/hermes-agent/gateway/run_turn_runner_clarify_delivery.py:93-100``).
For the Linear platform the native send SUCCEEDS — the plugin posts a real ``elicitation``
activity — but core then raises ``stale_generation`` from ``register_clarify``
(``gateway/run_turn_runner.py:544-551`` -> ``gateway/session_authority.py:959-962``), so the
plain-text fallback renders ``"❓ <question>"`` (``gateway/platforms/base.py:2901-2917``) into the
plugin's ``send()`` and posts a ``response`` activity. Linear marks the session complete on that
response, so the user never gets to answer (ENG-2222/ENG-2223).

These tests drive the real ``TurnRunner._clarify_callback_sync`` seam with the real plugin
adapter; only core's mutable store is faked.
"""

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import pytest

# ``gateway.run`` pulls in ``hermes_bootstrap``, which re-execs the current command inside the
# Hermes store venv when a source-update tail looks pending; that venv has no pytest, so the
# relaunched process dies. This suite drives core's TurnRunner from the plugin's own standalone
# venv, so opt out of the relaunch (harness plumbing only — nothing under test reads it).
if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS", "").strip().lower() not in ("1", "true", "yes", "on"):
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"

from gateway.config import PlatformConfig
from hermes_state_runtime import RuntimeStoreError
from tools import clarify_gateway as cm

from hermes_linear_agent.adapter import LinearAgentAdapter

SESSION_KEY = "sk-clarify-fallback"
ISSUE_ID = "issue-1"
AGENT_SESSION_ID = "session-1"
PICK = [{"qid": "q0", "question": "Which package should change?", "choices": ["frontend", "backend"], "multi_select": False}]


class _RecordingLinearClient:
    """Records every Linear mutation as a logical activity."""

    def __init__(self):
        self.elicitation_calls = []
        self.response_calls = []
        self.comment_calls = []

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
        record = {
            "agent_session_id": agent_session_id,
            "body": body,
            "response_type": response_type,
            "signal": signal,
            "signal_metadata": signal_metadata,
        }
        if response_type == "elicitation":
            self.elicitation_calls.append(record)
            return {"id": f"elicitation-{len(self.elicitation_calls)}"}
        self.response_calls.append(record)
        return {"id": f"response-{len(self.response_calls)}"}

    async def create_comment(self, issue_id, body, *, parent_id=None, mutation_policy=None):
        self.comment_calls.append({"issue_id": issue_id, "body": body})
        return {"id": f"comment-{len(self.comment_calls)}"}

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        return {"id": "thought-1"}

    async def create_error(self, agent_session_id, body):
        return {"id": "error-1"}

    async def list_issue_agent_sessions(self, issue_id, limit=50):
        return []


class _StaleGenerationAuthority:
    """``register_clarify`` fails exactly like ``gateway/session_authority.py:959-962``.

    The elicitation is already posted by the time core registers the prompt, which is the
    staleness race that makes core classify the (successful) send as ``failed``.
    """

    def __init__(self):
        self.sessions = {
            AGENT_SESSION_ID: SimpleNamespace(controls=SimpleNamespace(snapshot=lambda *a, **k: None)),
        }

    def register_clarify(self, session_id, generation, entry, *, timeout_s=None):
        raise RuntimeStoreError("stale_generation")


def _make_adapter(client):
    return LinearAgentAdapter(
        PlatformConfig(
            enabled=True,
            extra={"access_token": "linear-token", "webhook_secret": "secret", "allow_all_users": True},
        ),
        client=client,
    )


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def _runner(adapter, loop, monkeypatch, timeout=5):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id=ISSUE_ID,
        _status_thread_metadata=None,
        session_key=SESSION_KEY,
        stream_consumer_holder=[None],
        _loop_for_step=loop,
    )
    runner._close_native_stream_boundary = lambda *a, **k: None
    runner._approval_owner = (_StaleGenerationAuthority(), AGENT_SESSION_ID, 7)
    monkeypatch.setattr(cm, "get_clarify_timeout", lambda: timeout)
    return runner


def _wait_until(predicate, *, deadline=5.0):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _drive_core_clarify(adapter, runner, answer):
    """Run one clarify through core's real lane, answer it, and return the tool reply."""
    send_finished = threading.Event()
    fallback_texts = []
    real_send = adapter.send

    async def _spy_send(chat_id, content, reply_to=None, metadata=None):
        try:
            return await real_send(chat_id, content, reply_to=reply_to, metadata=metadata)
        finally:
            fallback_texts.append(content)
            send_finished.set()

    adapter.send = _spy_send
    reply = {}
    worker = threading.Thread(target=lambda: reply.update(runner._clarify_callback_sync(PICK)), daemon=True)
    worker.start()

    assert _wait_until(lambda: adapter._client.elicitation_calls), "the plugin never posted the elicitation"
    assert send_finished.wait(5), "core never ran its text fallback"
    # The failed-send fallback already ran; the question is still open for the user's answer.
    pending = cm.get_pending_for_session(SESSION_KEY, include_choice_prompts=True)
    assert pending is not None and pending.awaiting_text

    cm.resolve_text_response_for_session(SESSION_KEY, answer)
    worker.join(timeout=10)
    assert not worker.is_alive()
    return reply, fallback_texts


def test_one_clarify_posts_one_elicitation_and_no_text_fallback(loop, monkeypatch):
    """Live bug: one clarify produced an ``elicitation`` AND a plain-text ``response``."""
    client = _RecordingLinearClient()
    adapter = _make_adapter(client)
    adapter._latest_agent_session[ISSUE_ID] = AGENT_SESSION_ID
    runner = _runner(adapter, loop, monkeypatch)

    reply, fallback_texts = _drive_core_clarify(adapter, runner, "backend")

    assert len(fallback_texts) == 1  # core really did try the text replay
    elicitation = client.elicitation_calls[0]
    assert len(client.elicitation_calls) == 1
    assert elicitation["response_type"] == "elicitation"
    assert elicitation["signal"] == "select"
    assert elicitation["signal_metadata"] == {
        "options": [{"label": "frontend", "value": "frontend"}, {"label": "backend", "value": "backend"}]
    }
    # The regression: core's "❓ ..." fallback text reached Linear as a response activity.
    assert client.response_calls == [], (
        "core's text clarify fallback was posted as a Linear response activity: "
        f"{[call['body'] for call in client.response_calls]}"
    )
    assert client.comment_calls == []
    assert reply == {"answers": {"q0": "backend"}, "outcome": "submitted"}


@pytest.mark.parametrize(("answer", "expected"), [("backend", "backend"), ("1", "frontend"), ("ship it", "ship it")])
def test_answer_resolution_still_works_after_the_fallback_is_suppressed(loop, monkeypatch, answer, expected):
    """Tom's reply — option value, option number, or free text — still resolves the clarify."""
    client = _RecordingLinearClient()
    adapter = _make_adapter(client)
    adapter._latest_agent_session[ISSUE_ID] = AGENT_SESSION_ID

    reply, _ = _drive_core_clarify(adapter, _runner(adapter, loop, monkeypatch), answer)

    assert client.response_calls == []
    assert reply == {"answers": {"q0": expected}, "outcome": "submitted"}


