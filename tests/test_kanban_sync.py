"""Tests for the kanban <-> Linear sync loops.

Ports the bridge's issuesync/webhook routing, watch/boardwatch and reconcile
cases. ``kanban_cli`` is monkeypatched at module level; Linear is a fake client
whose ``execute`` answers by GraphQL operation name.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from hermes_linear_agent import kanban_cli
from hermes_linear_agent.kanban_rules import Card
from hermes_linear_agent.kanban_sync import KanbanSync

BOARD = "peak-agent-tools"

STATES: list[tuple[str, str, str]] = [
    ("Triage", "s-triage", "triage"),
    ("Backlog", "s-backlog", "backlog"),
    ("Todo", "s-todo", "unstarted"),
    ("Working", "s-working", "started"),
    ("Reviewing", "s-reviewing", "started"),
    ("Blocked", "s-blocked", "started"),
    ("Done", "s-done", "completed"),
    ("Canceled", "s-canceled", "canceled"),
    ("Duplicate", "s-duplicate", "duplicate"),
]
STATE_ID = {name: state_id for name, state_id, _type in STATES}
STATE_TYPE = {state_id: state_type for _name, state_id, state_type in STATES}


def state(name: str) -> dict[str, str]:
    for found, state_id, state_type in STATES:
        if found == name:
            return {"id": state_id, "name": name, "type": state_type}
    raise AssertionError(f"no test state {name}")


def card(card_id: str, status: str, assignee: str | None = None, completed_at: int | None = None) -> Card:
    return Card(id=card_id, status=status, assignee=assignee, completed_at=completed_at)


def pr_link(number: int, status: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"number": number}
    if status is not None:
        metadata["status"] = status
    return {
        "url": f"https://github.com/Peak-Digital-Software/peak-agent-tools/pull/{number}",
        "sourceType": "github",
        "metadata": metadata,
    }


REPO_LINK = {
    "url": "https://github.com/Peak-Digital-Software/peak-agent-tools",
    "sourceType": "api",
    "metadata": {},
}


class FakeClient:
    """A LinearClient whose ``execute`` answers canned GraphQL operations."""

    def __init__(
        self,
        *,
        current: dict[str, str] | None = None,
        states: list[tuple[str, str, str]] | None = None,
        attachments: dict[str, list[dict[str, Any]]] | None = None,
        fail_attachments: Exception | None = None,
        fail_update: Exception | None = None,
        active_pages: dict[str | None, dict[str, Any]] | None = None,
        fail_active: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.current = {key: state(name) for key, name in (current or {}).items()}
        self.states = states if states is not None else STATES
        self.attachments = attachments or {}
        self.fail_attachments = fail_attachments
        self.fail_update = fail_update
        self.active_pages = active_pages or {}
        self.fail_active = fail_active
        self.delay = delay
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.updates: list[dict[str, str]] = []
        self.active_calls = 0
        self.max_in_flight = 0
        self._in_flight = 0

    def operations(self, name: str) -> list[dict[str, Any]]:
        return [variables for query, variables in self.calls if query.startswith(f"query {name}(")]

    def _issue_key(self, issue_id: str) -> str | None:
        for key, node in self.current.items():
            if node["id"] == issue_id:
                return key
        return None

    async def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        variables = variables or {}
        self.calls.append((query, variables))

        if query.startswith("query WorkflowState("):
            state_id = str(variables["id"])
            return {"workflowState": {"id": state_id, "type": STATE_TYPE.get(state_id)}}

        if query.startswith("query IssueAttachments("):
            if self.fail_attachments is not None:
                raise self.fail_attachments
            return {"issue": {"attachments": {"nodes": self.attachments.get(str(variables["id"]), [])}}}

        if query.startswith("query TeamStates("):
            nodes = [{"id": state_id, "name": name, "type": state_type} for name, state_id, state_type in self.states]
            return {"team": {"states": {"nodes": nodes}}}

        if query.startswith("query Issue("):
            key = str(variables["id"])
            node = self.current.get(key) or state("Todo")
            return {
                "issue": {
                    "id": f"issue-{key}",
                    "identifier": key,
                    "state": node,
                    "team": {"id": "team-eng", "key": "ENG"},
                }
            }

        if query.startswith("mutation IssueUpdate("):
            if self.fail_update is not None:
                raise self.fail_update
            issue_id = str(variables["id"])
            state_id = str(variables["stateId"])
            self.updates.append({"id": issue_id, "stateId": state_id})
            key = self._issue_key(issue_id)
            if key is not None:
                for name, sid, state_type in self.states:
                    if sid == state_id:
                        self.current[key] = {"id": sid, "name": name, "type": state_type}
                        break
            return {"issueUpdate": {"success": True}}

        if query.startswith("query ActiveIssues("):
            self.active_calls += 1
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            try:
                if self.delay:
                    await asyncio.sleep(self.delay)
                if self.fail_active is not None:
                    raise self.fail_active
                page = self.active_pages.get(variables.get("after"), EMPTY_PAGE)
                return {"issues": page}
            finally:
                self._in_flight -= 1

        raise AssertionError(f"unexpected query: {query.splitlines()[0]}")


EMPTY_PAGE: dict[str, Any] = {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}


class Boards:
    """The kanban CLI plane, in memory."""

    def __init__(self) -> None:
        self.slugs: list[str] = [BOARD]
        self.cards_by: dict[tuple[str, str], list[Card]] = {}
        self.shows: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.cursors: dict[str, int] = {}
        self.boards_calls = 0
        self.archive_calls: list[tuple[str, list[str]]] = []
        self.saved: list[tuple[str, int]] = []

    def rows(self) -> list[dict[str, Any]]:
        return [{"slug": slug, "db_path": f"/tmp/{slug}.db"} for slug in self.slugs]

    async def boards(self) -> list[dict[str, Any]]:
        self.boards_calls += 1
        return self.rows()

    async def cards(self, board: str, key: str) -> list[Card]:
        return list(self.cards_by.get((board, key), []))

    async def show(self, board: str, task_id: str) -> dict[str, Any]:
        return dict(self.shows.get(task_id, {"id": task_id, "tenant": None, "completed": False}))

    async def archive(self, board: str, ids: list[str]) -> None:
        self.archive_calls.append((board, list(ids)))
        for card_key, cards in self.cards_by.items():
            if card_key[0] != board:
                continue
            self.cards_by[card_key] = [
                Card(card.id, "archived", card.assignee, card.completed_at) if card.id in ids else card
                for card in cards
            ]

    def read_events(self, db_path: str, after: int, limit: int = 200) -> list[dict[str, Any]]:
        board = db_path.removeprefix("/tmp/").removesuffix(".db")
        return [row for row in self.events.get(board, []) if row["id"] > after][:limit]

    def max_event_id(self, db_path: str) -> int:
        board = db_path.removeprefix("/tmp/").removesuffix(".db")
        return max((row["id"] for row in self.events.get(board, [])), default=0)

    def load_cursor(self, board: str) -> int | None:
        return self.cursors.get(board)

    def save_cursor(self, board: str, event_id: int) -> None:
        self.cursors[board] = event_id
        self.saved.append((board, event_id))


@pytest.fixture
def boards(monkeypatch) -> Boards:
    fake = Boards()
    monkeypatch.setattr(kanban_cli, "boards", fake.boards)
    monkeypatch.setattr(kanban_cli, "cards", fake.cards)
    monkeypatch.setattr(kanban_cli, "show", fake.show)
    monkeypatch.setattr(kanban_cli, "archive", fake.archive)
    monkeypatch.setattr(kanban_cli, "read_events", fake.read_events)
    monkeypatch.setattr(kanban_cli, "max_event_id", fake.max_event_id)
    monkeypatch.setattr(kanban_cli, "load_cursor", fake.load_cursor)
    monkeypatch.setattr(kanban_cli, "save_cursor", fake.save_cursor)
    return fake


async def until(ready, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not ready():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting for the condition")
        await asyncio.sleep(0.002)


def event(event_id: int, task_id: str, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": event_id, "task_id": task_id, "kind": kind, "payload": payload or {}, "tenant": None}


# -- webhook -> close cards ----------------------------------------------------


def issue_payload(
    identifier: str,
    *,
    action: str = "update",
    was_state: str | None = None,
    now_state: str | None = None,
    kind: str = "Issue",
) -> dict[str, Any]:
    data: dict[str, Any] = {"identifier": identifier}
    if now_state is not None:
        data["stateId"] = now_state
    payload: dict[str, Any] = {"type": kind, "action": action, "data": data}
    if was_state is not None:
        payload["updatedFrom"] = {"stateId": was_state}
    return payload


async def test_unhandled_payload_type_returns_an_ignore_line(boards):
    sync = KanbanSync(FakeClient())

    line = await sync.handle_issue_webhook({"type": "Comment", "action": "create"})

    assert line == "route -: ignored: unhandled Comment/create"
    assert boards.boards_calls == 0


async def test_create_names_no_previous_state_and_is_ignored(boards):
    sync = KanbanSync(FakeClient())

    line = await sync.handle_issue_webhook(issue_payload("ENG-1", action="create", now_state="s-unstarted"))

    assert line == "route ENG-1: ignored: no card action"
    assert boards.archive_calls == []


async def test_update_that_never_left_triage_is_ignored(boards):
    sync = KanbanSync(FakeClient())

    line = await sync.handle_issue_webhook(issue_payload("ENG-1", was_state="s-triage", now_state="s-triage"))

    assert line == "route ENG-1: ignored: no card action"


@pytest.mark.parametrize(
    ("was", "now", "expected"),
    [
        ("s-triage", "s-canceled", "ignored: no card action"),
        ("s-triage", "s-duplicate", "ignored: no card action"),
        ("s-done", "s-backlog", "ignored: no card action"),
        ("s-canceled", "s-backlog", "ignored: no card action"),
    ],
)
async def test_non_moves_create_no_card_action(boards, was, now, expected):
    sync = KanbanSync(FakeClient())

    line = await sync.handle_issue_webhook(issue_payload("ENG-1", was_state=was, now_state=now))

    assert line == f"route ENG-1: {expected}"


async def test_canceled_move_archives_the_open_cards(boards):
    sync = KanbanSync(FakeClient())
    boards.cards_by[(BOARD, "ENG-1")] = [
        card("t_open", "ready"),
        card("t_done", "done"),
        card("t_gone", "archived"),
    ]

    line = await sync.handle_issue_webhook(issue_payload("ENG-1", was_state="s-working", now_state="s-canceled"))

    assert line == "route ENG-1: cancel: archived 1"
    assert boards.archive_calls == [(BOARD, ["t_open"])]


async def test_done_by_hand_archives_and_a_duplicate_closes_like_canceled(boards):
    sync = KanbanSync(FakeClient())
    boards.cards_by[(BOARD, "ENG-2")] = [card("t_2", "ready")]
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "blocked")]

    close = await sync.handle_issue_webhook(issue_payload("ENG-2", was_state="s-working", now_state="s-done"))
    duplicate = await sync.handle_issue_webhook(issue_payload("ENG-3", was_state="s-working", now_state="s-duplicate"))

    assert close == "route ENG-2: close: archived 1"
    assert duplicate == "route ENG-3: cancel: archived 1"


async def test_an_echo_with_no_open_cards_makes_no_archive_call(boards):
    sync = KanbanSync(FakeClient())
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_done", "done"), card("t_gone", "archived")]

    line = await sync.handle_issue_webhook(issue_payload("ENG-1", was_state="s-working", now_state="s-done"))

    assert line == "route ENG-1: close: archived 0"
    assert boards.archive_calls == []


async def test_close_cards_retries_a_board_that_never_settles_three_times(boards, monkeypatch):
    sync = KanbanSync(FakeClient())
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_a", "running")]
    calls: list[tuple[str, list[str]]] = []

    async def never_settles(board: str, ids: list[str]) -> None:
        calls.append((board, list(ids)))

    monkeypatch.setattr(kanban_cli, "archive", never_settles)

    archived = await sync._close_cards("ENG-1")

    assert archived == 3
    assert calls == [(BOARD, ["t_a"]), (BOARD, ["t_a"]), (BOARD, ["t_a"])]


async def test_close_cards_archives_every_open_card_on_every_board_in_one_call(boards):
    sync = KanbanSync(FakeClient())
    boards.slugs = ["board-a", "board-b"]
    boards.cards_by[("board-a", "ENG-1")] = [card("t_a", "ready"), card("t_b", "running")]
    boards.cards_by[("board-b", "ENG-1")] = [card("t_c", "blocked")]

    assert await sync._close_cards("ENG-1") == 3

    assert boards.archive_calls == [("board-a", ["t_a", "t_b"]), ("board-b", ["t_c"])]


async def test_a_state_type_is_resolved_once_per_process(boards):
    client = FakeClient()
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "ready")]

    for _ in range(2):
        await sync.handle_issue_webhook(issue_payload("ENG-1", was_state="s-working", now_state="s-done"))

    # both state ids were looked up exactly once
    assert [variables["id"] for variables in client.operations("WorkflowState")] == ["s-working", "s-done"]


# -- card events -> issue status ----------------------------------------------


async def test_a_canceled_move_to_a_card_whose_tenant_is_not_an_issue_key_does_nothing(boards):
    sync = KanbanSync(FakeClient())
    boards.shows["t_other"] = {"id": "t_other", "tenant": "not-a-key", "completed": False}

    await sync._handle_event(BOARD, event(1, "t_other", "claimed"))

    assert boards.boards_calls == 0


async def test_heartbeat_is_noise_and_never_calls_the_cli(boards):
    sync = KanbanSync(FakeClient())

    await sync._handle_event(BOARD, event(1, "t_root", "heartbeat"))

    assert boards.boards_calls == 0


async def test_non_done_status_sets_working_even_when_attachments_would_fail(boards):
    client = FakeClient(current={"ENG-1": "Todo"}, fail_attachments=RuntimeError("would be a bug"))
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "running", "coder")]

    await sync._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-working"}]


async def test_no_status_is_written_when_it_already_matches(boards, caplog):
    client = FakeClient(current={"ENG-1": "Working"})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "running", "coder")]

    with caplog.at_level(logging.INFO):
        await sync._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert client.updates == []
    assert "status Working" not in caplog.text


async def test_a_status_change_logs_the_acted_line_before_the_write(boards, caplog):
    client = FakeClient(current={"ENG-1": "Todo"})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "running", "coder")]

    with caplog.at_level(logging.INFO):
        await sync._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert f"{BOARD} t_1 claimed -> ENG-1 status Working" in caplog.text
    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-working"}]


async def test_a_blocked_card_moves_the_issue_to_blocked(boards):
    client = FakeClient(current={"ENG-1": "Working"})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "blocked", "coder")]

    await sync._handle_event(BOARD, event(1, "t_1", "blocked"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-blocked"}]


async def test_an_unblock_returns_to_the_card_set_status(boards):
    client = FakeClient(current={"ENG-1": "Blocked"})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "ready", "coder")]

    await sync._handle_event(BOARD, event(1, "t_1", "unblocked"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-working"}]


async def test_a_team_without_blocked_state_is_a_silent_no_op(boards, caplog):
    no_blocked = [node for node in STATES if node[0] != "Blocked"]
    client = FakeClient(current={"ENG-1": "Working"}, states=no_blocked)
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "blocked", "coder")]

    with caplog.at_level(logging.INFO):
        await sync._handle_event(BOARD, event(1, "t_1", "blocked"))

    assert client.updates == []
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


async def test_d7_a_card_event_never_writes_to_a_closed_issue(boards):
    client = FakeClient(current={"ENG-1": "Working"})
    client.current["ENG-1"] = {"id": "s-done", "name": "Working", "type": "completed"}
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "done", "coder", 1), card("t_2", "done", "reviewer", 2)]

    await sync._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == []


async def test_an_open_pr_holds_a_completed_card_set_in_reviewing(boards):
    client = FakeClient(
        current={"ENG-1": "Working"},
        attachments={"issue-ENG-1": [REPO_LINK, pr_link(156, "open")]},
    )
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": True}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "done", "coder", 1), card("t_2", "done", "reviewer", 2)]

    await sync._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-reviewing"}]


async def test_a_merged_pr_lets_the_completed_card_set_reach_done(boards):
    client = FakeClient(current={"ENG-1": "Reviewing"}, attachments={"issue-ENG-1": [pr_link(156, "merged")]})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": True}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "done", "coder", 1), card("t_2", "done", "reviewer", 2)]

    await sync._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-done"}]


async def test_an_attachment_fetch_failure_holds_done_in_reviewing_with_one_warning(boards, caplog):
    client = FakeClient(current={"ENG-1": "Working"}, fail_attachments=RuntimeError("attachments unavailable"))
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": True}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "done", "coder", 1), card("t_2", "done", "reviewer", 2)]

    with caplog.at_level(logging.INFO):
        await sync._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-reviewing"}]
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == [
        f"warn: watch {BOARD} t_1: attachments unavailable, holding ENG-1 in Reviewing: attachments unavailable"
    ]


# -- followers and cursors ----------------------------------------------------


async def test_a_cold_start_seeds_the_cursor_from_the_newest_event(boards):
    sync = KanbanSync(FakeClient())
    boards.events[BOARD] = [event(7, "t_old", "claimed"), event(9, "t_new", "claimed")]

    cursor = await sync._seed_cursor(BOARD, f"/tmp/{BOARD}.db")

    assert cursor == 9
    assert boards.saved == [(BOARD, 9)]


async def test_a_saved_cursor_is_resumed_without_seeding(boards):
    sync = KanbanSync(FakeClient())
    boards.cursors[BOARD] = 4

    assert await sync._seed_cursor(BOARD, f"/tmp/{BOARD}.db") == 4
    assert boards.saved == []


async def test_the_follower_saves_the_cursor_after_each_event(boards):
    sync = KanbanSync(FakeClient(), poll_seconds=0.01)
    boards.cursors[BOARD] = 0
    boards.shows["t_1"] = {"id": "t_1", "tenant": None, "completed": False}
    boards.events[BOARD] = [event(1, "t_1", "claimed"), event(2, "t_2", "heartbeat")]

    task = asyncio.create_task(sync._follow_forever(BOARD, f"/tmp/{BOARD}.db"))
    try:
        await until(lambda: boards.cursors.get(BOARD) == 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert boards.saved == [(BOARD, 1), (BOARD, 2)]


async def test_a_failing_event_logs_and_the_follower_keeps_going(boards, caplog, monkeypatch):
    sync = KanbanSync(FakeClient(), poll_seconds=0.01)
    boards.cursors[BOARD] = 0
    boards.events[BOARD] = [event(1, "t_boom", "claimed"), event(2, "t_ok", "claimed")]
    original_show = boards.show

    async def show(board, task_id):
        if task_id == "t_boom":
            raise RuntimeError("show exploded")
        return await original_show(board, task_id)

    monkeypatch.setattr(kanban_cli, "show", show)
    boards.shows["t_ok"] = {"id": "t_ok", "tenant": None, "completed": False}

    with caplog.at_level(logging.ERROR):
        task = asyncio.create_task(sync._follow_forever(BOARD, f"/tmp/{BOARD}.db"))
        try:
            await until(lambda: boards.cursors.get(BOARD) == 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert "show exploded" in caplog.text
    assert boards.cursors[BOARD] == 2


async def test_board_discovery_starts_exactly_one_follower_per_new_slug(boards):
    sync = KanbanSync(FakeClient(), poll_seconds=0.01)
    boards.slugs = ["a"]
    boards.cursors["a"] = 0

    assert await sync._poll_boards() == ["a"]
    boards.slugs = ["a", "b"]
    boards.cursors["b"] = 0
    assert await sync._poll_boards() == ["b"]
    assert await sync._poll_boards() == []
    assert [task for task in sync._followers if not task.done()]

    for task in sync._followers:
        task.cancel()
    await asyncio.gather(*sync._followers, return_exceptions=True)


async def test_start_returns_the_discovery_and_reconcile_tasks(boards):
    sync = KanbanSync(FakeClient(), discovery_seconds=60, reconcile_seconds=60, poll_seconds=0.01)
    boards.cursors[BOARD] = 0

    tasks = sync.start()
    try:
        await until(lambda: BOARD in sync._started_boards)
        assert len(tasks) == 2
    finally:
        for task in [*tasks, *sync._followers]:
            task.cancel()
        await asyncio.gather(*tasks, *sync._followers, return_exceptions=True)


# -- reconcile ----------------------------------------------------------------


def active_issue(key: str, state_name: str, attachments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": f"issue-{key}",
        "identifier": key,
        "title": f"Issue {key}",
        "description": "Body",
        "url": f"https://linear.app/peak/issue/{key}",
        "state": state(state_name),
        "team": {"id": "team-eng", "key": "ENG"},
        "attachments": {"nodes": attachments or []},
    }


def page(nodes: list[dict[str, Any]], has_next: bool = False, cursor: str | None = None) -> dict[str, Any]:
    return {"nodes": nodes, "pageInfo": {"hasNextPage": has_next, "endCursor": cursor}}


async def test_active_issues_follows_every_page_and_passes_string_literals(boards):
    client = FakeClient(
        active_pages={
            None: page([active_issue("ENG-1", "Backlog"), active_issue("ENG-2", "Working")], True, "cursor-1"),
            "cursor-1": page([active_issue("ENG-3", "Todo")]),
        }
    )
    sync = KanbanSync(client)

    found = await sync._active_issues()

    assert [node["identifier"] for node in found] == ["ENG-1", "ENG-2", "ENG-3"]
    assert [variables["after"] for variables in client.operations("ActiveIssues")] == [None, "cursor-1"]
    assert client.operations("ActiveIssues")[0]["team"] == "ENG"
    assert 'in: ["backlog", "unstarted", "started"]' in client.calls[0][0]
    assert "attachments(first: 10) { nodes { url sourceType metadata } }" in client.calls[0][0]


async def test_reconcile_leaves_an_issue_with_no_cards_alone(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-12", "Backlog")])})
    sync = KanbanSync(client)

    assert await sync.reconcile() == 0
    assert client.updates == []


async def test_reconcile_fixes_a_stale_status_then_is_silent_once_it_matches(boards, caplog):
    client = FakeClient(active_pages={None: page([active_issue("ENG-14", "Working")])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-14")] = [card("t_review", "review", "reviewer")]

    with caplog.at_level(logging.INFO):
        assert await sync.reconcile() == 1
    assert client.updates == [{"id": "issue-ENG-14", "stateId": "s-reviewing"}]
    assert "reconcile ENG-14: Working -> Reviewing" in caplog.text

    # the next pass sees the issue already in Reviewing
    client.active_pages[None] = page([active_issue("ENG-14", "Reviewing")])
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert await sync.reconcile() == 0
    assert client.updates == [{"id": "issue-ENG-14", "stateId": "s-reviewing"}]
    assert "reconcile ENG-14" not in caplog.text


async def test_reconcile_writes_blocked_while_any_card_is_blocked(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-17", "Working")])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-17")] = [card("t_a", "blocked", "coder")]

    assert await sync.reconcile() == 1
    assert client.updates == [{"id": "issue-ENG-17", "stateId": "s-blocked"}]


async def test_reconcile_holds_done_for_an_open_pr(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-18", "Working", [pr_link(156, "open")])])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-18")] = [card("t_a", "done", "coder", 1), card("t_b", "done", "reviewer", 2)]

    assert await sync.reconcile() == 1
    assert client.updates == [{"id": "issue-ENG-18", "stateId": "s-reviewing"}]


async def test_reconcile_writes_done_with_no_pr(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-20", "Working", [REPO_LINK])])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-20")] = [card("t_a", "done", "coder", 1), card("t_b", "done", "reviewer", 2)]

    assert await sync.reconcile() == 1
    assert client.updates == [{"id": "issue-ENG-20", "stateId": "s-done"}]


async def test_reconcile_loads_a_team_s_states_at_most_once_per_pass(boards):
    client = FakeClient(
        active_pages={
            None: page([active_issue("ENG-31", "Backlog"), active_issue("ENG-32", "Backlog")]),
        }
    )
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-31")] = [card("t_a", "running", "coder")]
    boards.cards_by[(BOARD, "ENG-32")] = [card("t_b", "running", "coder")]

    assert await sync.reconcile() == 2
    assert len(client.operations("TeamStates")) == 1


async def test_reconcile_concatenates_cards_across_every_board(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-40", "Working")])})
    sync = KanbanSync(client)
    boards.slugs = ["board-a", "board-b"]
    boards.cards_by[("board-a", "ENG-40")] = [card("t_a", "done", "coder", 1)]
    boards.cards_by[("board-b", "ENG-40")] = [card("t_b", "ready", "coder")]

    assert await sync.reconcile() == 0
    assert client.updates == []


async def test_reconcile_loop_logs_one_failure_line_and_survives(boards, caplog):
    client = FakeClient(fail_active=RuntimeError("graphql exploded"))
    sync = KanbanSync(client, reconcile_seconds=0.01)

    with caplog.at_level(logging.ERROR):
        task = asyncio.create_task(sync._reconcile_loop())
        try:
            await until(lambda: client.active_calls >= 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    failures = [record.getMessage() for record in caplog.records if record.getMessage().startswith("reconcile failed")]
    assert failures and failures[0] == "reconcile failed: graphql exploded"
    assert client.active_calls >= 2, "the loop kept going after the failed pass"


async def test_reconcile_runs_at_once_and_never_overlaps_a_slow_pass(boards):
    client = FakeClient(delay=0.02)
    sync = KanbanSync(client, reconcile_seconds=0.005)

    task = asyncio.create_task(sync._reconcile_loop())
    try:
        await until(lambda: client.active_calls >= 2)
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert client.max_in_flight == 1
    assert client.active_calls >= 2
