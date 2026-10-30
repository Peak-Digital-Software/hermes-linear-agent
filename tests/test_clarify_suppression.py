"""P17 unit level: ``send()`` drops core's plain-text replay of a posted elicitation.

The suppression predicate is only correct when every one of its conditions holds — the
``is_approval_prompt`` metadata flag, the ``❓ `` prefix, the recorded question for the same
chat, and a recent timestamp — because the exec-approval text fallback carries the same
metadata flag (`~/.hermes/hermes-agent/gateway/run_turn_runner.py:773`).
"""

import pytest
from gateway.config import PlatformConfig

from hermes_linear_agent import adapter as adapter_module
from hermes_linear_agent.adapter import CLARIFY_FALLBACK_SUPPRESS_SECONDS, LinearAgentAdapter

SESSION_KEY = "sk-clarify-suppression"
ISSUE_ID = "issue-1"
AGENT_SESSION_ID = "session-1"
QUESTION = "Which package should change?"
APPROVAL_METADATA = {"is_approval_prompt": True}


class _RecordingLinearClient:
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
        record = {"agent_session_id": agent_session_id, "body": body, "response_type": response_type}
        if response_type == "elicitation":
            self.elicitation_calls.append(record)
        else:
            self.response_calls.append(record)
        return {"id": f"activity-{len(self.elicitation_calls) + len(self.response_calls)}"}

    async def create_comment(self, issue_id, body, *, parent_id=None, mutation_policy=None):
        self.comment_calls.append({"issue_id": issue_id, "body": body})
        return {"id": "comment-1"}

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        return {"id": "thought-1"}


def _make_adapter(client):
    return LinearAgentAdapter(
        PlatformConfig(
            enabled=True,
            extra={"access_token": "linear-token", "webhook_secret": "secret", "allow_all_users": True},
        ),
        client=client,
    )


@pytest.fixture
def clock(monkeypatch):
    """A settable ``_clarify_now`` clock (never a global ``time.monotonic`` monkeypatch)."""
    now = [500.0]
    monkeypatch.setattr(adapter_module, "_clarify_now", lambda: now[0])
    return now


def _fallback_text(question=QUESTION):
    return (
        f"❓ {question}\nReply \"skip\" to skip this question.\n\n"
        "  1. frontend\n  2. backend\n\nReply with the number, the option text, or your own answer."
    )


async def _clarify_posted_adapter(client=None, question=QUESTION):
    client = client or _RecordingLinearClient()
    adapter = _make_adapter(client)
    adapter._latest_agent_session[ISSUE_ID] = AGENT_SESSION_ID
    result = await adapter.send_clarify(ISSUE_ID, question, ["frontend", "backend"], "clarify-1", SESSION_KEY)
    assert result.success is True
    assert adapter._clarify_elicitations[ISSUE_ID][0] == question
    return adapter, client


async def test_send_suppresses_the_replay_inside_the_window(clock):
    adapter, client = await _clarify_posted_adapter()

    result = await adapter.send(ISSUE_ID, _fallback_text(), metadata=APPROVAL_METADATA)

    assert result.success is True
    assert client.response_calls == []
    assert client.comment_calls == []
    assert adapter._clarify_elicitations == {}


async def test_send_lets_the_replay_through_after_the_window(clock):
    adapter, client = await _clarify_posted_adapter()

    clock[0] += CLARIFY_FALLBACK_SUPPRESS_SECONDS + 1
    result = await adapter.send(ISSUE_ID, _fallback_text(), metadata=APPROVAL_METADATA)

    assert result.success is True
    assert [call["body"] for call in client.response_calls] == [_fallback_text()]
    assert adapter._clarify_elicitations == {}


async def test_replay_for_another_chat_is_not_suppressed(clock):
    client = _RecordingLinearClient()
    adapter, _ = await _clarify_posted_adapter(client)
    adapter._latest_agent_session["issue-2"] = "session-2"

    await adapter.send("issue-2", _fallback_text(), metadata=APPROVAL_METADATA)

    assert [call["agent_session_id"] for call in client.response_calls] == ["session-2"]


async def test_exec_approval_shaped_text_is_not_suppressed(clock):
    """The exec-approval fallback also sets is_approval_prompt; only the clarify text matches."""
    adapter, client = await _clarify_posted_adapter()

    await adapter.send(ISSUE_ID, "Approve this command?\n\n1. Allow\n2. Deny", metadata=APPROVAL_METADATA)

    assert len(client.response_calls) == 1


async def test_replay_of_a_different_question_is_not_suppressed(clock):
    adapter, client = await _clarify_posted_adapter()

    await adapter.send(ISSUE_ID, _fallback_text("Delete the branch?"), metadata=APPROVAL_METADATA)

    assert len(client.response_calls) == 1


async def test_replay_without_the_approval_flag_is_not_suppressed(clock):
    adapter, client = await _clarify_posted_adapter()

    await adapter.send(ISSUE_ID, _fallback_text(), metadata=None)

    assert len(client.response_calls) == 1


async def test_a_second_identical_replay_is_sent(clock):
    """One suppression per elicitation: a genuine second prompt must not be dropped."""
    adapter, client = await _clarify_posted_adapter()

    suppressed = await adapter.send(ISSUE_ID, _fallback_text(), metadata=APPROVAL_METADATA)
    second = await adapter.send(ISSUE_ID, _fallback_text(), metadata=APPROVAL_METADATA)

    assert suppressed.success is True
    assert second.success is True
    assert len(client.response_calls) == 1
