"""Linear tools accept the argument shapes the model actually sends (ENG-2292).

Hermes core rejects a deferred ``tool_call`` missing a schema ``required``
name before the handler runs, so no schema may emit ``required``; handlers own
the missing-argument errors.
"""

import pytest

from hermes_linear_agent import tools as t
from hermes_linear_agent.adapter import LinearAgentAdapter
from hermes_linear_agent.client import LinearGraphQLClient
from hermes_linear_agent.registry import set_active_adapter

ISSUE_UUID = "11111111-2222-3333-4444-555555555555"
TEAM_ID = "team-1"
SESSION_ID = "sess-newest"
ISSUE_CHAT_ID = "issue-chat-1"


class _FakeClient(LinearGraphQLClient):
    def __init__(self):
        self.calls = []

    async def get_issue(self, issue_id):
        self.calls.append(("get_issue", issue_id))
        if issue_id in ("ENG-2285", ISSUE_UUID):
            return {"id": ISSUE_UUID, "identifier": "ENG-2285", "team": {"id": TEAM_ID}}
        return None

    async def get_issue_status(self, name=None, team=None):
        self.calls.append(("get_issue_status", name, team))
        return {"id": "state-working", "name": name}

    async def update_issue(self, issue_id, input_data, mutation_policy=None):
        self.calls.append(("update_issue", issue_id, dict(input_data)))
        return {"id": issue_id}

    async def list_comments(self, issue_id=None, limit=50):
        self.calls.append(("list_comments", issue_id, limit))
        return []

    async def update_session_plan(self, session_id, plan):
        self.calls.append(("update_session_plan", session_id, plan))
        return {"id": session_id}

    async def set_agent_session_external_urls(self, session_id, urls):
        self.calls.append(("set_agent_session_external_urls", session_id, urls))
        return {"id": session_id}

    async def create_response(self, session_id, body, **kwargs):
        self.calls.append(("create_response", session_id))
        return {"id": "act-1"}


def _adapter():
    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakeClient()
    adapter._mutation_policy = None
    adapter._latest_agent_session = {ISSUE_CHAT_ID: SESSION_ID}
    set_active_adapter(adapter)
    return adapter


@pytest.fixture
def linear_session_env():
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="linear_agent", chat_id=ISSUE_CHAT_ID)
    yield
    clear_session_vars(tokens)


@pytest.fixture
def no_session_env():
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars()
    yield
    clear_session_vars(tokens)


def _mutations(client):
    return [c for c in client.calls if c[0] == "update_issue"]


PLAN = [{"content": "Do it", "status": "inProgress"}]


# ── schemas never emit ``required`` ──────────────────────────────────────────


def test_no_registered_schema_emits_required():
    offenders = [schema["name"] for _h, schema, _e in t._TOOLS if "required" in schema["parameters"]]
    assert offenders == []


@pytest.mark.parametrize(
    "name",
    [
        "linear_agent_list_comments",
        "linear_agent_update_plan",
        "linear_agent_update_issue",
        "linear_agent_set_session_links",
        "linear_agent_request_auth",
    ],
)
def test_core_validator_accepts_tool_without_required(name):
    schema = next(s for _h, s, _e in t._TOOLS if s["name"] == name)
    assert "required" not in schema["parameters"]


# ── the shapes the model sends ───────────────────────────────────────────────


async def test_list_comments_accepts_camel_case_issue_id():
    adapter = _adapter()
    await t.linear_agent_list_comments(issueId="ENG-2285")
    assert adapter._client.calls == [("list_comments", "ENG-2285", 50)]


async def test_update_plan_defaults_to_current_session(linear_session_env):
    adapter = _adapter()
    result = await t.linear_agent_update_plan(plan=PLAN)
    assert result.startswith("✅")
    assert adapter._client.calls == [("update_session_plan", SESSION_ID, PLAN)]


async def test_explicit_session_id_wins_over_env(linear_session_env):
    adapter = _adapter()
    await t.linear_agent_update_plan(agent_session_id="explicit", plan=PLAN)
    assert adapter._client.calls[0][1] == "explicit"


async def test_set_session_links_defaults_to_current_session(linear_session_env):
    adapter = _adapter()
    result = await t.linear_agent_set_session_links(links=["https://example.com/pr/1"])
    assert result.startswith("✅")
    assert adapter._client.calls[0][1] == SESSION_ID


async def test_request_auth_defaults_to_current_session(linear_session_env):
    adapter = _adapter()
    result = await t.linear_agent_request_auth(url="https://example.com/auth")
    assert result.startswith("✅")
    assert adapter._client.calls == [("create_response", SESSION_ID)]


async def test_update_issue_flat_id_and_state():
    adapter = _adapter()
    result = await t.linear_agent_update_issue(id=ISSUE_UUID, state="Working")
    assert result.startswith("✅")
    assert _mutations(adapter._client) == [("update_issue", ISSUE_UUID, {"stateId": "state-working"})]


async def test_update_issue_identifier_resolves_to_uuid_with_one_fetch():
    adapter = _adapter()
    result = await t.linear_agent_update_issue(id="ENG-2285", input={"state": "Working"})
    assert result.startswith("✅")
    assert _mutations(adapter._client) == [("update_issue", ISSUE_UUID, {"stateId": "state-working"})]
    assert [c for c in adapter._client.calls if c[0] == "get_issue"] == [("get_issue", "ENG-2285")]


async def test_update_issue_unknown_identifier_is_not_found():
    adapter = _adapter()
    result = await t.linear_agent_update_issue(id="ENG-9999", input={"title": "x"})
    assert result == "❌ linear_agent_update_issue: issue ENG-9999 not found"
    assert _mutations(adapter._client) == []


# ── truly missing arguments still error in the handler ───────────────────────


async def test_update_issue_empty_is_handler_error():
    adapter = _adapter()
    result = await t.linear_agent_update_issue()
    assert result.startswith("❌ linear_agent_update_issue: missing")
    assert adapter._client.calls == []


async def test_update_plan_without_session_env_is_handler_error(no_session_env):
    adapter = _adapter()
    result = await t.linear_agent_update_plan(plan=PLAN)
    assert result.startswith("❌ linear_agent_update_plan: missing agent_session_id")
    assert adapter._client.calls == []
