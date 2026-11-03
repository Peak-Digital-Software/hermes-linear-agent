"""Tests for the kanban <-> Linear sync loops.

Ports the bridge's issuesync/webhook routing, watch/boardwatch and reconcile
cases. ``kanban_cli`` is monkeypatched at module level; Linear is a fake client
whose ``execute`` answers by GraphQL operation name. The filesystem watcher is
injectable (``kanban_sync._awatch``) so the loops can be driven by hand; the
boards-root and board-db watches are also exercised against real temp dirs.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import watchfiles

from hermes_linear_agent import kanban_cli, kanban_sync
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
    ("Ready to Merge", "s-ready-to-merge", "started"),
    ("Done", "s-done", "completed"),
    ("Canceled", "s-canceled", "canceled"),
    ("Duplicate", "s-duplicate", "duplicate"),
]
OPS_STATES: list[tuple[str, str, str]] = [
    ("Triage", "o-triage", "triage"),
    ("Backlog", "o-backlog", "backlog"),
    ("Todo", "o-todo", "unstarted"),
    ("In Progress", "o-in-progress", "started"),
    ("In Review", "o-in-review", "started"),
    ("Blocked", "o-blocked", "started"),
    ("Waiting on client", "o-waiting", "started"),
    ("Done", "o-done", "completed"),
    ("Canceled", "o-canceled", "canceled"),
    ("Duplicate", "o-duplicate", "duplicate"),
]
STATE_ID = {name: state_id for name, state_id, _type in STATES}
STATE_TYPE = {state_id: state_type for _name, state_id, state_type in STATES + OPS_STATES}


def state(name: str, team: str = "ENG") -> dict[str, str]:
    for found, state_id, state_type in OPS_STATES if team == "OPS" else STATES:
        if found == name:
            return {"id": state_id, "name": name, "type": state_type}
    raise AssertionError(f"no test state {name}")


def team_of(identifier: str) -> dict[str, str]:
    key = next((k for k in ("OPS", "ENB") if identifier.startswith(f"{k}-")), "ENG")
    return {"id": f"team-{key.lower()}", "key": key}


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
        issue_states: dict[str, dict[str, str]] | None = None,
        parents: dict[str, dict[str, str]] | None = None,
        children: dict[str, list[dict[str, Any]]] | None = None,
        ops_pages: dict[str | None, dict[str, Any]] | None = None,
        updated_at: dict[str, str] | None = None,
    ) -> None:
        # issue key -> updatedAt, for the hand-set guard (ENG-2378)
        self.updated_at = updated_at or {}
        self.current = {key: state(name, team_of(key)["key"]) for key, name in (current or {}).items()}
        # ActiveIssues pages for the OPS team; `active_pages` answers ENG
        self.ops_pages = ops_pages or {}
        self.states = states if states is not None else STATES
        self.attachments = attachments or {}
        self.fail_attachments = fail_attachments
        self.fail_update = fail_update
        self.active_pages = active_pages or {}
        self.fail_active = fail_active
        self.delay = delay
        # issue id ("issue-ENG-1") -> state, for issues that are not in `current`
        self.issue_states = issue_states or {}
        # issue key or id -> {"id": ..., "identifier": ...}
        self.parents = parents or {}
        # parent key or id -> child nodes
        self.children = children or {}
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
            states = OPS_STATES if variables.get("teamId") in {"team-ops", "team-enb"} else self.states
            nodes = [{"id": state_id, "name": name, "type": state_type} for name, state_id, state_type in states]
            return {"team": {"states": {"nodes": nodes}}}

        if query.startswith("query Issue("):
            key = str(variables["id"])
            issue_id = key if key.startswith("issue-") else f"issue-{key}"
            identifier = key.removeprefix("issue-") if key.startswith("issue-") else key
            team = team_of(identifier)
            node = self.current.get(identifier) or self.issue_states.get(issue_id) or state("Todo", team["key"])
            return {
                "issue": {
                    "id": issue_id,
                    "identifier": identifier,
                    "updatedAt": self.updated_at.get(identifier),
                    "state": node,
                    "team": team,
                    "parent": self.parents.get(identifier) or self.parents.get(issue_id),
                    "children": {"nodes": self.children.get(issue_id) or self.children.get(identifier) or []},
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
                for name, sid, state_type in self.states + OPS_STATES:
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
                team = variables.get("team")
                pages = {"ENG": self.active_pages, "OPS": self.ops_pages}.get(team, {})
                page = pages.get(variables.get("after"), EMPTY_PAGE)
                return {"issues": page}
            finally:
                self._in_flight -= 1

        raise AssertionError(f"unexpected query: {query.splitlines()[0]}")


EMPTY_PAGE: dict[str, Any] = {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}


class Boards:
    """The kanban CLI plane, in memory."""

    def __init__(self) -> None:
        self.slugs: list[str] = [BOARD]
        self.archived: set[str] = set()
        self.paths: dict[str, str] = {}
        self.cards_by: dict[tuple[str, str], list[Card]] = {}
        self.done_whens: dict[str, str | None] = {}
        self.shows: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.cursors: dict[str, int] = {}
        self.boards_calls = 0
        self.archive_calls: list[tuple[str, list[str]]] = []
        self.saved: list[tuple[str, int]] = []

    def db_path(self, slug: str) -> str:
        return self.paths.get(slug, f"/tmp/{slug}.db")

    def rows(self) -> list[dict[str, Any]]:
        return [{"slug": slug, "db_path": self.db_path(slug), "archived": slug in self.archived} for slug in self.slugs]

    async def boards(self) -> list[dict[str, Any]]:
        self.boards_calls += 1
        return self.rows()

    async def cards(self, board: str, key: str) -> list[Card]:
        return list(self.cards_by.get((board, key), []))

    async def show(self, board: str, task_id: str) -> dict[str, Any]:
        return dict(self.shows.get(task_id, {"id": task_id, "tenant": None, "completed": False}))

    async def done_when(self, board: str, task_id: str) -> str | None:
        return self.done_whens.get(task_id)

    async def archive(self, board: str, ids: list[str]) -> None:
        self.archive_calls.append((board, list(ids)))
        for card_key, cards in self.cards_by.items():
            if card_key[0] != board:
                continue
            self.cards_by[card_key] = [
                Card(card.id, "archived", card.assignee, card.completed_at) if card.id in ids else card
                for card in cards
            ]

    def _slug_for(self, db_path: str) -> str:
        for slug, path in self.paths.items():
            if path == db_path:
                return slug
        return db_path.removeprefix("/tmp/").removesuffix(".db")

    def read_events(self, db_path: str, after: int, limit: int = 200) -> list[dict[str, Any]]:
        board = self._slug_for(db_path)
        return [row for row in self.events.get(board, []) if row["id"] > after][:limit]

    def max_event_id(self, db_path: str) -> int:
        board = self._slug_for(db_path)
        return max((row["id"] for row in self.events.get(board, [])), default=0)

    def load_cursor(self, board: str, db_path: str) -> int | None:
        return self.cursors.get(board)

    def save_cursor(self, board: str, db_path: str, event_id: int) -> None:
        self.cursors[board] = event_id
        self.saved.append((board, event_id))


@pytest.fixture
def cli(monkeypatch) -> Boards:
    """The CLI plane's boards/cards/show/archive, in memory."""
    fake = Boards()
    monkeypatch.setattr(kanban_cli, "boards", fake.boards)
    monkeypatch.setattr(kanban_cli, "cards", fake.cards)
    monkeypatch.setattr(kanban_cli, "show", fake.show)
    monkeypatch.setattr(kanban_cli, "done_when", fake.done_when)
    monkeypatch.setattr(kanban_cli, "archive", fake.archive)
    return fake


@pytest.fixture
def boards(cli, monkeypatch) -> Boards:
    """The CLI plane plus its event reads and cursors, in memory."""
    monkeypatch.setattr(kanban_cli, "read_events", cli.read_events)
    monkeypatch.setattr(kanban_cli, "max_event_id", cli.max_event_id)
    monkeypatch.setattr(kanban_cli, "load_cursor", cli.load_cursor)
    monkeypatch.setattr(kanban_cli, "save_cursor", cli.save_cursor)
    return cli


class FakeWatch:
    """An injectable ``awatch``: every ``yield_change`` releases one batch."""

    def __init__(self) -> None:
        self.queues: list[asyncio.Queue[None]] = []

    def __call__(self, *paths, **kwargs):
        queue: asyncio.Queue[None] = asyncio.Queue()
        self.queues.append(queue)
        return self._batches(queue)

    @staticmethod
    async def _batches(queue: asyncio.Queue[None]):
        while True:
            await queue.get()
            yield set()

    def yield_change(self) -> None:
        for queue in self.queues:
            queue.put_nowait(None)


@pytest.fixture
def watch(monkeypatch) -> FakeWatch:
    fake = FakeWatch()
    monkeypatch.setattr(kanban_sync, "_awatch", fake)
    return fake


@pytest.fixture
def idle_watch(monkeypatch):
    """A watcher that never fires: the follower is driven by its initial read."""

    async def never(*paths, **kwargs):
        await asyncio.Event().wait()
        yield set()  # pragma: no cover

    monkeypatch.setattr(kanban_sync, "_awatch", never)
    return never


@pytest.fixture
def watching(monkeypatch) -> asyncio.Event:
    """The real watcher, wrapped so a test can wait until a watch has started."""
    started = asyncio.Event()
    real = watchfiles.awatch

    async def wrapped(*paths, **kwargs):
        started.set()
        async for changes in real(*paths, **kwargs):
            yield changes

    monkeypatch.setattr(kanban_sync, "_awatch", wrapped)
    return started


@pytest.fixture
def state_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("PEAK_AGENT_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


async def until(ready, timeout: float = 3.0) -> None:
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
    parent_id: str | None = None,
    parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"identifier": identifier}
    if now_state is not None:
        data["stateId"] = now_state
    if parent_id is not None:
        data["parentId"] = parent_id
    if parent is not None:
        data["parent"] = parent
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


async def test_a_close_keeps_a_running_card_and_archives_the_rest(boards):
    sync = KanbanSync(FakeClient())
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_run", "running"), card("t_ready", "ready")]

    line = await sync.handle_issue_webhook(issue_payload("ENG-1", was_state="s-working", now_state="s-done"))

    assert line == "route ENG-1: close: archived 1"
    assert boards.archive_calls == [(BOARD, ["t_ready"])]


async def test_a_cancel_still_archives_a_running_card(boards):
    sync = KanbanSync(FakeClient())
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_run", "running"), card("t_ready", "ready")]

    line = await sync.handle_issue_webhook(issue_payload("ENG-1", was_state="s-working", now_state="s-canceled"))

    assert line == "route ENG-1: cancel: archived 2"
    assert boards.archive_calls == [(BOARD, ["t_run", "t_ready"])]


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


# -- child Done -> wake the parent --------------------------------------------


async def test_a_child_moving_to_done_wakes_the_parent_session(boards):
    woke: list[tuple[str, str, str]] = []
    sync = KanbanSync(FakeClient(), wake=lambda *args: woke.append(args))
    boards.cards_by[(BOARD, "ENG-2")] = [card("t_2", "ready")]

    payload = issue_payload(
        "ENG-2",
        was_state="s-working",
        now_state="s-done",
        parent_id="issue-ENG-1",
        parent={"id": "issue-ENG-1", "identifier": "ENG-1"},
    )
    line = await sync.handle_issue_webhook(payload)

    assert line == "route ENG-2: close: archived 1"
    assert woke == [("issue-ENG-1", "ENG-1", "[sync] ENG-2 is Done.")]


async def test_a_child_done_without_an_embedded_parent_fetches_its_key_once(boards):
    woke: list[tuple[str, str, str]] = []
    client = FakeClient()
    sync = KanbanSync(client, wake=lambda *args: woke.append(args))
    boards.cards_by[(BOARD, "ENG-2")] = [card("t_2", "ready")]

    await sync.handle_issue_webhook(
        issue_payload("ENG-2", was_state="s-working", now_state="s-done", parent_id="issue-ENG-1")
    )

    assert woke == [("issue-ENG-1", "ENG-1", "[sync] ENG-2 is Done.")]
    assert [variables["id"] for variables in client.operations("Issue")] == ["issue-ENG-1"]


async def test_without_a_wake_callable_a_child_done_touches_no_parent(boards):
    client = FakeClient()
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-2")] = [card("t_2", "ready")]

    line = await sync.handle_issue_webhook(
        issue_payload("ENG-2", was_state="s-working", now_state="s-done", parent_id="issue-ENG-1")
    )

    assert line == "route ENG-2: close: archived 1"
    assert client.operations("Issue") == []


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

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-todo"}]


async def test_a_todo_card_set_writes_todo(boards):
    client = FakeClient(current={"ENG-1": "Working"})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "todo", "coder"), card("t_2", "ready", "coder")]

    await sync._handle_event(BOARD, event(1, "t_1", "created"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-todo"}]


async def test_a_review_card_writes_reviewing_whatever_its_assignee(boards):
    client = FakeClient(current={"ENG-1": "Working"})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "review", "oracle")]

    await sync._handle_event(BOARD, event(1, "t_1", "review_requested"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-reviewing"}]


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


async def test_an_open_pr_holds_a_completed_card_set_in_ready_to_merge(boards):
    client = FakeClient(
        current={"ENG-1": "Working"},
        attachments={"issue-ENG-1": [REPO_LINK, pr_link(156, "open")]},
    )
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": True}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "done", "coder", 1), card("t_2", "done", "reviewer", 2)]

    await sync._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-ready-to-merge"}]


async def test_a_merged_pr_lets_the_completed_card_set_reach_done(boards):
    client = FakeClient(current={"ENG-1": "Ready to Merge"}, attachments={"issue-ENG-1": [pr_link(156, "merged")]})
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": True}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "done", "coder", 1), card("t_2", "done", "reviewer", 2)]

    await sync._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-done"}]


async def test_an_attachment_fetch_failure_holds_done_in_ready_to_merge_with_one_warning(boards, caplog):
    client = FakeClient(current={"ENG-1": "Working"}, fail_attachments=RuntimeError("attachments unavailable"))
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": True}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "done", "coder", 1), card("t_2", "done", "reviewer", 2)]

    with caplog.at_level(logging.INFO):
        await sync._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-ready-to-merge"}]
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == [
        f"warn: watch {BOARD} t_1: attachments unavailable, holding ENG-1 in Ready to Merge: attachments unavailable"
    ]


# -- parent rollup -------------------------------------------------------------


def child_node(key: str, state_name: str) -> dict[str, Any]:
    return {"id": f"issue-{key}", "identifier": key, "state": {"name": state_name}}


async def test_a_blocked_child_rolls_the_parent_up_to_blocked(boards):
    client = FakeClient(current={"ENG-2": "Working", "ENG-3": "Working"})
    client.issue_states["issue-ENG-1"] = state("Working")
    client.children["issue-ENG-1"] = [child_node("ENG-2", "Working"), child_node("ENG-3", "Working")]
    client.parents["ENG-3"] = {"id": "issue-ENG-1", "identifier": "ENG-1"}
    sync = KanbanSync(client)
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": False}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "blocked", "coder")]

    await sync._handle_event(BOARD, event(1, "t_3", "blocked"))

    assert client.updates == [
        {"id": "issue-ENG-3", "stateId": "s-blocked"},
        {"id": "issue-ENG-1", "stateId": "s-blocked"},
    ]


async def test_a_working_child_rolls_the_parent_up_to_working(boards):
    client = FakeClient(current={"ENG-3": "Todo"})
    client.issue_states["issue-ENG-1"] = state("Blocked")
    # the just-written child state is taken from the write, not from this read
    client.children["issue-ENG-1"] = [child_node("ENG-2", "Done"), child_node("ENG-3", "Todo")]
    client.parents["ENG-3"] = {"id": "issue-ENG-1", "identifier": "ENG-1"}
    sync = KanbanSync(client)
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": False}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "running", "coder")]

    await sync._handle_event(BOARD, event(1, "t_3", "claimed"))

    assert client.updates == [
        {"id": "issue-ENG-3", "stateId": "s-working"},
        {"id": "issue-ENG-1", "stateId": "s-working"},
    ]


async def test_a_parent_whose_children_are_all_done_is_left_alone(boards):
    client = FakeClient(current={"ENG-3": "Todo"})
    client.issue_states["issue-ENG-1"] = state("Working")
    client.children["issue-ENG-1"] = [child_node("ENG-2", "Done"), child_node("ENG-3", "Todo")]
    client.parents["ENG-3"] = {"id": "issue-ENG-1", "identifier": "ENG-1"}
    sync = KanbanSync(client)
    boards.done_whens["t_3"] = "pass"
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": True}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "done", "coder", 1)]

    await sync._handle_event(BOARD, event(1, "t_3", "completed"))

    assert client.updates == [{"id": "issue-ENG-3", "stateId": "s-done"}]


async def test_a_parent_already_at_the_rollup_is_not_rewritten(boards):
    client = FakeClient(current={"ENG-3": "Todo"})
    client.issue_states["issue-ENG-1"] = state("Working")
    client.children["issue-ENG-1"] = [child_node("ENG-2", "Working"), child_node("ENG-3", "Todo")]
    client.parents["ENG-3"] = {"id": "issue-ENG-1", "identifier": "ENG-1"}
    sync = KanbanSync(client)
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": False}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "running", "coder")]

    await sync._handle_event(BOARD, event(1, "t_3", "claimed"))

    assert client.updates == [{"id": "issue-ENG-3", "stateId": "s-working"}]


async def test_an_event_on_an_issue_with_children_writes_the_rollup_not_its_cards(boards):
    client = FakeClient(current={"ENG-1": "Working"})
    client.children["ENG-1"] = [child_node("ENG-2", "Working")]
    sync = KanbanSync(client)
    boards.shows["t_1"] = {"id": "t_1", "tenant": "ENG-1", "completed": False}
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_1", "review", "oracle")]

    await sync._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert client.updates == []


async def test_a_work_issue_without_a_parent_never_fetches_one(boards):
    client = FakeClient(current={"ENG-3": "Todo"})
    sync = KanbanSync(client)
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": False}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "running", "coder")]

    await sync._handle_event(BOARD, event(1, "t_3", "claimed"))

    assert client.updates == [{"id": "issue-ENG-3", "stateId": "s-working"}]
    assert [variables["id"] for variables in client.operations("Issue")] == ["ENG-3", "ENG-3"]


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


async def test_the_follower_reads_once_on_start_and_saves_the_cursor_after_each_event(boards, idle_watch):
    sync = KanbanSync(FakeClient())
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


async def test_the_follower_watches_only_the_db_folder_not_its_subtree(boards, monkeypatch):
    # The default board's db is ~/.hermes/kanban.db: a recursive watch on ~/.hermes exhausts inotify.
    seen: list[tuple[tuple, dict]] = []

    async def record(*paths, **kwargs):
        seen.append((paths, kwargs))
        await asyncio.Event().wait()
        yield set()  # pragma: no cover

    monkeypatch.setattr(kanban_sync, "_awatch", record)
    sync = KanbanSync(FakeClient())
    boards.cursors[BOARD] = 0
    task = asyncio.create_task(sync._follow_forever(BOARD, f"/tmp/{BOARD}.db"))
    try:
        await until(lambda: bool(seen))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    paths, kwargs = seen[0]
    assert paths == ("/tmp",)
    assert kwargs["recursive"] is False


async def test_a_filesystem_change_reads_the_events_that_landed_after_the_start(boards, watch):
    sync = KanbanSync(FakeClient())
    boards.cursors[BOARD] = 0
    boards.shows["t_1"] = {"id": "t_1", "tenant": None, "completed": False}

    task = asyncio.create_task(sync._follow_forever(BOARD, f"/tmp/{BOARD}.db"))
    try:
        await until(lambda: bool(watch.queues))
        boards.events[BOARD] = [event(1, "t_1", "claimed")]
        watch.yield_change()
        await until(lambda: boards.cursors.get(BOARD) == 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert boards.saved == [(BOARD, 1)]


async def test_a_failing_event_logs_and_the_follower_keeps_going(boards, caplog, monkeypatch):
    sync = KanbanSync(FakeClient())
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


async def test_a_vanished_db_stops_the_follower_with_no_error_log(boards, caplog, monkeypatch):
    sync = KanbanSync(FakeClient())
    boards.cursors[BOARD] = 5

    def gone(db_path: str, after: int, limit: int = 200) -> list[dict[str, Any]]:
        raise FileNotFoundError(db_path)

    monkeypatch.setattr(kanban_cli, "read_events", gone)

    with caplog.at_level(logging.DEBUG):
        await asyncio.wait_for(sync._follow_forever(BOARD, f"/tmp/{BOARD}.db"), timeout=2)

    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []


async def test_a_read_failure_logs_one_line_and_never_retries(boards, caplog, monkeypatch):
    sync = KanbanSync(FakeClient())
    boards.cursors[BOARD] = 5
    calls: list[str] = []

    def locked(db_path: str, after: int, limit: int = 200) -> list[dict[str, Any]]:
        calls.append(db_path)
        raise RuntimeError("database is locked")

    monkeypatch.setattr(kanban_cli, "read_events", locked)

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(sync._follow_forever(BOARD, f"/tmp/{BOARD}.db"), timeout=2)

    assert calls == [f"/tmp/{BOARD}.db"]
    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert errors == [f"error: watch {BOARD}: database is locked"]


async def test_board_discovery_starts_one_follower_per_new_slug_and_cancels_the_rest(boards, idle_watch):
    sync = KanbanSync(FakeClient())
    boards.slugs = ["a"]
    boards.cursors["a"] = 0

    assert await sync._poll_boards() == ["a"]
    boards.slugs = ["a", "b"]
    boards.cursors["b"] = 0
    assert await sync._poll_boards() == ["b"]
    assert await sync._poll_boards() == []
    assert set(sync._followers) == {"a", "b"}

    boards.slugs = ["a"]
    assert await sync._poll_boards() == []
    assert set(sync._followers) == {"a"}

    followers = list(sync._followers.values())
    for follower in followers:
        follower.cancel()
    await asyncio.gather(*followers, return_exceptions=True)


async def test_an_archived_board_is_never_followed(boards, idle_watch):
    sync = KanbanSync(FakeClient())
    boards.slugs = ["a", "b"]
    boards.archived = {"b"}

    assert await sync._poll_boards() == ["a"]
    assert set(sync._followers) == {"a"}

    followers = list(sync._followers.values())
    for follower in followers:
        follower.cancel()
    await asyncio.gather(*followers, return_exceptions=True)


async def test_start_returns_the_discovery_and_reconcile_tasks(boards, idle_watch, tmp_path):
    sync = KanbanSync(FakeClient(), boards_root=tmp_path / "boards", reconcile_seconds=60)
    boards.cursors[BOARD] = 0

    tasks = sync.start()
    try:
        await until(lambda: BOARD in sync._followers)
        assert len(tasks) == 2
    finally:
        followers = list(sync._followers.values())
        for task in [*tasks, *followers]:
            task.cancel()
        await asyncio.gather(*tasks, *followers, return_exceptions=True)


async def test_start_adds_the_pr_loop_only_when_pr_gh_is_set(boards, idle_watch, tmp_path, monkeypatch):
    async def noop(**_kwargs):
        return []

    monkeypatch.setattr("hermes_linear_agent.pr_reconcile.reconcile_prs", noop)
    sync = KanbanSync(FakeClient(), boards_root=tmp_path / "boards", reconcile_seconds=60, pr_gh="/bin/gh")
    boards.cursors[BOARD] = 0

    tasks = sync.start()
    try:
        await until(lambda: BOARD in sync._followers)
        assert len(tasks) == 3
    finally:
        followers = list(sync._followers.values())
        for task in [*tasks, *followers]:
            task.cancel()
        await asyncio.gather(*tasks, *followers, return_exceptions=True)


async def test_the_pr_loop_survives_a_raising_pass(monkeypatch):
    calls = []

    async def flaky(*, gh):
        calls.append(gh)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return []

    monkeypatch.setattr("hermes_linear_agent.pr_reconcile.reconcile_prs", flaky)
    sync = KanbanSync(FakeClient(), pr_gh="/bin/gh", pr_seconds=0)
    task = asyncio.create_task(sync._pr_loop())
    try:
        await until(lambda: len(calls) >= 2)
        assert not task.done()
        assert calls[:2] == ["/bin/gh", "/bin/gh"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_cancelling_the_discovery_task_cancels_its_followers(boards, idle_watch, tmp_path):
    sync = KanbanSync(FakeClient(), boards_root=tmp_path / "boards", reconcile_seconds=60)
    boards.cursors[BOARD] = 0

    discovery, reconcile = sync.start()
    await until(lambda: BOARD in sync._followers)
    followers = list(sync._followers.values())
    assert followers
    discovery.cancel()
    reconcile.cancel()
    await asyncio.gather(discovery, reconcile, *followers, return_exceptions=True)

    assert all(task.done() for task in followers)


# -- real filesystem watches ---------------------------------------------------


def make_board_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, assignee TEXT, tenant TEXT);"
        "CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, kind TEXT, payload TEXT);"
    )
    conn.commit()
    conn.close()


def add_board_event(path: Path, task_id: str = "t_1", kind: str = "claimed") -> None:
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO task_events (task_id, kind, payload) VALUES (?, ?, '{}')", (task_id, kind))
    conn.commit()
    conn.close()


async def test_a_board_recreated_under_the_same_slug_receives_its_first_event(cli, watching, state_dir, tmp_path):
    root = tmp_path / "boards"
    folder = root / "eng"
    folder.mkdir(parents=True)
    db = folder / "kanban.db"
    make_board_db(db)
    kanban_cli.save_cursor("eng", str(db), 5)  # the first incarnation's cursor

    # the board is removed and re-created under the same slug: a new inode
    # Hold the old file open so its inode stays allocated: ext4 (CI runners) reuses a freed inode at once.
    with open(db, "rb"):
        db.unlink()
        make_board_db(db)
    assert kanban_cli.load_cursor("eng", str(db)) is None, "a new inode starts from scratch"
    cli.paths["eng"] = str(db)

    sync = KanbanSync(FakeClient(), boards_root=root)
    task = asyncio.create_task(sync._follow_forever("eng", str(db)))
    try:
        await asyncio.wait_for(watching.wait(), timeout=2)
        await until(lambda: kanban_cli.load_cursor("eng", str(db)) == 0)
        add_board_event(db)
        await until(lambda: kanban_cli.load_cursor("eng", str(db)) == 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert kanban_cli.load_cursor("eng", str(db)) == 1


async def test_a_removed_board_follower_exits_with_no_error_log(cli, watching, state_dir, tmp_path, caplog):
    root = tmp_path / "boards"
    folder = root / "eng"
    folder.mkdir(parents=True)
    db = folder / "kanban.db"
    make_board_db(db)
    cli.paths["eng"] = str(db)

    sync = KanbanSync(FakeClient(), boards_root=root)
    with caplog.at_level(logging.DEBUG):
        task = asyncio.create_task(sync._follow_forever("eng", str(db)))
        try:
            await asyncio.wait_for(watching.wait(), timeout=2)
            await until(lambda: kanban_cli.load_cursor("eng", str(db)) == 0)
            shutil.rmtree(folder)
            await asyncio.wait_for(task, timeout=5)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert task.done()
    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []


# -- reconcile ----------------------------------------------------------------


def active_issue(
    key: str,
    state_name: str,
    attachments: list[dict[str, Any]] | None = None,
    children: list[dict[str, Any]] | None = None,
    parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": f"issue-{key}",
        "identifier": key,
        "title": f"Issue {key}",
        "description": "Body",
        "url": f"https://linear.app/peak/issue/{key}",
        "state": state(state_name, team_of(key)["key"]),
        "team": team_of(key),
        "attachments": {"nodes": attachments or []},
    }
    if children is not None:
        node["children"] = {"nodes": children}
    if parent is not None:
        node["parent"] = parent
    return node


def page(nodes: list[dict[str, Any]], has_next: bool = False, cursor: str | None = None) -> dict[str, Any]:
    return {"nodes": nodes, "pageInfo": {"hasNextPage": has_next, "endCursor": cursor}}


async def test_active_issues_follows_every_page_and_passes_string_literals(boards):
    client = FakeClient(
        active_pages={
            None: page([active_issue("ENG-1", "Backlog"), active_issue("ENG-2", "Working")], True, "cursor-1"),
            "cursor-1": page([active_issue("ENG-3", "Todo")]),
        }
    )
    sync = KanbanSync(client, team_keys=("ENG",))

    found = await sync._active_issues()

    assert [node["identifier"] for node in found] == ["ENG-1", "ENG-2", "ENG-3"]
    assert [variables["after"] for variables in client.operations("ActiveIssues")] == [None, "cursor-1"]
    assert client.operations("ActiveIssues")[0]["team"] == "ENG"
    assert 'in: ["backlog", "unstarted", "started"]' in client.calls[0][0]
    assert "attachments(first: 10) { nodes { url sourceType metadata } }" in client.calls[0][0]
    assert "parent { id identifier }" in client.calls[0][0]
    assert "children { nodes { id identifier state { name } } }" in client.calls[0][0]


async def test_active_issues_flattens_the_parent_and_children(boards):
    node = active_issue(
        "ENG-1",
        "Working",
        children=[child_node("ENG-2", "Done")],
        parent={"id": "issue-ENG-9", "identifier": "ENG-9"},
    )
    client = FakeClient(active_pages={None: page([node])})
    sync = KanbanSync(client)

    found = await sync._active_issues()

    assert found[0]["parent"] == {"id": "issue-ENG-9", "identifier": "ENG-9"}
    assert found[0]["children"] == [{"id": "issue-ENG-2", "identifier": "ENG-2", "state": {"name": "Done"}}]


async def test_reconcile_leaves_an_issue_with_no_cards_alone(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-12", "Backlog")])})
    sync = KanbanSync(client)

    assert await sync.reconcile() == 0
    assert client.updates == []


async def test_reconcile_fixes_a_stale_status_then_is_silent_once_it_matches(boards, caplog):
    client = FakeClient(active_pages={None: page([active_issue("ENG-14", "Working")])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-14")] = [card("t_review", "review", "oracle")]

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
    assert client.updates == [{"id": "issue-ENG-18", "stateId": "s-ready-to-merge"}]


async def test_reconcile_writes_done_with_no_pr(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-20", "Working", [REPO_LINK])])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-20")] = [card("t_a", "done", "coder", 1), card("t_b", "done", "reviewer", 2)]
    boards.done_whens["t_b"] = "pass"

    assert await sync.reconcile() == 1
    assert client.updates == [{"id": "issue-ENG-20", "stateId": "s-done"}]


async def test_reconcile_writes_nothing_when_all_done_without_a_pr_or_evidence(boards):
    client = FakeClient(active_pages={None: page([active_issue("ENG-20", "Working", [REPO_LINK])])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-20")] = [card("t_a", "done", "coder", 1), card("t_b", "done", "reviewer", 2)]
    boards.done_whens["t_a"] = "fail"

    assert await sync.reconcile() == 0
    assert client.updates == []


async def test_an_event_with_all_done_and_no_evidence_writes_nothing(boards):
    client = FakeClient(current={"ENG-3": "Working"}, attachments={"issue-ENG-3": [REPO_LINK]})
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": True}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "done", "coder", 1)]

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_3", "completed"))

    assert client.updates == []
    assert not client.operations("IssueUpdate")


async def test_an_event_with_all_done_and_evidence_writes_done(boards):
    client = FakeClient(current={"ENG-3": "Working"}, attachments={"issue-ENG-3": []})
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": True}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "done", "coder", 1)]
    boards.done_whens["t_3"] = "pass"

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_3", "completed"))

    assert client.updates == [{"id": "issue-ENG-3", "stateId": "s-done"}]


def _iso_ago(seconds: float) -> str:
    import datetime as dt

    return (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


async def test_a_card_state_write_skips_an_issue_hand_set_five_minutes_ago(boards):
    client = FakeClient(current={"ENG-3": "Todo"}, updated_at={"ENG-3": _iso_ago(300)})
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": False}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "running", "coder")]

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_3", "claimed"))

    assert client.updates == []


async def test_a_merged_pr_done_writes_even_when_hand_set_five_minutes_ago(boards):
    client = FakeClient(
        current={"ENG-3": "Ready to Merge"},
        attachments={"issue-ENG-3": [pr_link(7, "merged")]},
        updated_at={"ENG-3": _iso_ago(300)},
    )
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": True}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "done", "coder", 1)]

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_3", "completed"))

    assert client.updates == [{"id": "issue-ENG-3", "stateId": "s-done"}]


async def test_an_edit_older_than_thirty_minutes_does_not_block_the_sync(boards):
    client = FakeClient(current={"ENG-3": "Todo"}, updated_at={"ENG-3": _iso_ago(31 * 60)})
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": False}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "running", "coder")]

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_3", "claimed"))

    assert client.updates == [{"id": "issue-ENG-3", "stateId": "s-working"}]


async def test_the_syncs_own_write_is_not_mistaken_for_a_hand_edit(boards):
    client = FakeClient(current={"ENG-3": "Todo"})
    sync = KanbanSync(client)
    boards.shows["t_3"] = {"id": "t_3", "tenant": "ENG-3", "completed": False}
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "running", "coder")]
    await sync._handle_event(BOARD, event(1, "t_3", "claimed"))
    assert len(client.updates) == 1

    # Linear stamps updatedAt a few seconds after our own write; the card moves on
    client.updated_at["ENG-3"] = _iso_ago(0)
    boards.cards_by[(BOARD, "ENG-3")] = [card("t_3", "review", "coder")]
    await sync._handle_event(BOARD, event(2, "t_3", "completed"))

    assert [u["stateId"] for u in client.updates] == ["s-working", "s-reviewing"]


async def test_reconcile_rolls_a_parent_up_from_its_children(boards):
    node = active_issue(
        "ENG-1",
        "Working",
        children=[child_node("ENG-2", "Blocked"), child_node("ENG-3", "Working")],
    )
    client = FakeClient(active_pages={None: page([node])})
    sync = KanbanSync(client)

    assert await sync.reconcile() == 1
    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-blocked"}]


async def test_reconcile_prefers_the_rollup_over_a_parent_s_own_cards(boards):
    node = active_issue("ENG-1", "Todo", children=[child_node("ENG-2", "Done")])
    client = FakeClient(active_pages={None: page([node])})
    sync = KanbanSync(client)
    boards.cards_by[(BOARD, "ENG-1")] = [card("t_a", "running", "coder")]

    assert await sync.reconcile() == 0
    assert client.updates == []


async def test_reconcile_leaves_a_parent_whose_children_are_all_done(boards):
    node = active_issue("ENG-1", "Working", children=[child_node("ENG-2", "Done"), child_node("ENG-3", "Canceled")])
    client = FakeClient(active_pages={None: page([node])})
    sync = KanbanSync(client)

    assert await sync.reconcile() == 0
    assert client.updates == []


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
    boards.cards_by[("board-b", "ENG-40")] = [card("t_b", "running", "coder")]

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


# -- OPS team: its own state names, and hand-set states are left alone -----------


def ops_event_setup(boards, key: str, card_status: str, assignee: str = "coder", **kwargs: Any) -> None:
    boards.shows["t_1"] = {"id": "t_1", "tenant": key, "completed": False}
    boards.cards_by[(BOARD, key)] = [card("t_1", card_status, assignee, **kwargs)]


async def test_an_ops_claimed_card_writes_in_progress(boards):
    client = FakeClient(current={"OPS-1": "Todo"})
    ops_event_setup(boards, "OPS-1", "running")

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert client.updates == [{"id": "issue-OPS-1", "stateId": "o-in-progress"}]


async def test_an_ops_review_writes_in_review(boards):
    client = FakeClient(current={"OPS-1": "In Progress"})
    ops_event_setup(boards, "OPS-1", "review", "oracle")

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_1", "review_requested"))

    assert client.updates == [{"id": "issue-OPS-1", "stateId": "o-in-review"}]


async def test_an_ops_blocked_card_writes_blocked(boards):
    client = FakeClient(current={"OPS-1": "In Progress"})
    ops_event_setup(boards, "OPS-1", "blocked")

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_1", "blocked"))

    assert client.updates == [{"id": "issue-OPS-1", "stateId": "o-blocked"}]


@pytest.mark.parametrize("attachments", [[], [pr_link(7, "merged")]])
async def test_ops_all_done_with_a_merged_or_no_pr_writes_done(boards, attachments):
    client = FakeClient(current={"OPS-1": "In Review"}, attachments={"issue-OPS-1": attachments})
    ops_event_setup(boards, "OPS-1", "done", completed_at=1)
    boards.done_whens["t_1"] = "pass"

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_1", "completed"))

    assert client.updates == [{"id": "issue-OPS-1", "stateId": "o-done"}]


async def test_an_ops_issue_already_in_its_state_is_not_rewritten(boards):
    client = FakeClient(current={"OPS-1": "In Progress"})
    ops_event_setup(boards, "OPS-1", "running")

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert client.updates == []


async def test_an_ops_issue_waiting_on_client_gets_no_event_write(boards):
    client = FakeClient(current={"OPS-1": "Waiting on client"})
    ops_event_setup(boards, "OPS-1", "running")

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert client.updates == []


async def test_waiting_on_client_is_only_hand_set_on_ops(boards):
    # an ENG team that happened to have such a state is not protected
    states = STATES + [("Waiting on client", "s-waiting", "started")]
    client = FakeClient(current={"ENG-1": "Todo"}, states=states)
    client.current["ENG-1"] = {"id": "s-waiting", "name": "Waiting on client", "type": "started"}
    ops_event_setup(boards, "ENG-1", "running")

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_1", "claimed"))

    assert client.updates == [{"id": "issue-ENG-1", "stateId": "s-working"}]


async def test_an_ops_parent_rolls_up_to_in_progress_when_a_child_goes_in_progress(boards):
    client = FakeClient(current={"OPS-3": "Todo"})
    client.issue_states["issue-OPS-1"] = state("Todo", "OPS")
    client.children["issue-OPS-1"] = [child_node("OPS-2", "Done"), child_node("OPS-3", "Todo")]
    client.parents["OPS-3"] = {"id": "issue-OPS-1", "identifier": "OPS-1"}
    boards.shows["t_3"] = {"id": "t_3", "tenant": "OPS-3", "completed": False}
    boards.cards_by[(BOARD, "OPS-3")] = [card("t_3", "running", "coder")]

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_3", "claimed"))

    assert client.updates == [
        {"id": "issue-OPS-3", "stateId": "o-in-progress"},
        {"id": "issue-OPS-1", "stateId": "o-in-progress"},
    ]


async def test_an_ops_parent_waiting_on_client_is_not_rolled_up(boards):
    client = FakeClient(current={"OPS-3": "Todo"})
    client.issue_states["issue-OPS-1"] = state("Waiting on client", "OPS")
    client.children["issue-OPS-1"] = [child_node("OPS-3", "Todo")]
    client.parents["OPS-3"] = {"id": "issue-OPS-1", "identifier": "OPS-1"}
    boards.shows["t_3"] = {"id": "t_3", "tenant": "OPS-3", "completed": False}
    boards.cards_by[(BOARD, "OPS-3")] = [card("t_3", "running", "coder")]

    await KanbanSync(client)._handle_event(BOARD, event(1, "t_3", "claimed"))

    assert client.updates == [{"id": "issue-OPS-3", "stateId": "o-in-progress"}]


async def test_reconcile_writes_an_ops_issue_with_its_own_state_names(boards):
    client = FakeClient(ops_pages={None: page([active_issue("OPS-5", "Todo")])})
    boards.cards_by[(BOARD, "OPS-5")] = [card("t_a", "running", "coder")]

    assert await KanbanSync(client).reconcile() == 1
    assert client.updates == [{"id": "issue-OPS-5", "stateId": "o-in-progress"}]


async def test_reconcile_skips_an_ops_issue_waiting_on_client(boards):
    client = FakeClient(ops_pages={None: page([active_issue("OPS-5", "Waiting on client")])})
    boards.cards_by[(BOARD, "OPS-5")] = [card("t_a", "running", "coder")]

    assert await KanbanSync(client).reconcile() == 0
    assert client.updates == []


async def test_reconcile_rolls_an_ops_parent_up_to_in_progress(boards):
    node = active_issue("OPS-1", "Todo", children=[child_node("OPS-2", "In Progress")])
    client = FakeClient(ops_pages={None: page([node])})

    assert await KanbanSync(client).reconcile() == 1
    assert client.updates == [{"id": "issue-OPS-1", "stateId": "o-in-progress"}]


async def test_active_issues_pages_every_team_key(boards):
    client = FakeClient(
        active_pages={None: page([active_issue("ENG-1", "Todo")])},
        ops_pages={
            None: page([active_issue("OPS-1", "Todo")], True, "c1"),
            "c1": page([active_issue("OPS-2", "Todo")]),
        },
    )

    found = await KanbanSync(client)._active_issues()

    assert [node["identifier"] for node in found] == ["ENG-1", "OPS-1", "OPS-2"]
    assert [(v["team"], v["after"]) for v in client.operations("ActiveIssues")] == [
        ("ENG", None),
        ("OPS", None),
        ("OPS", "c1"),
        ("ENB", None),
    ]


# -- ENG-2343: wake the delegated ancestor of an undelegated waiter ----------


def _waiting_issue(identifier: str, blocker_state: str, parent: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": f"issue-{identifier}",
        "identifier": identifier,
        "state": {"type": "unstarted"},
        "delegate": None,
        "inverseRelations": {
            "nodes": [
                {
                    "type": "blocks",
                    "issue": {"id": "issue-ENG-9", "identifier": "ENG-9", "state": {"type": blocker_state}},
                }
            ]
        },
        "parent": parent,
    }


class WaiterClient(FakeClient):
    """FakeClient that also answers the waiters' BlockedByMe query."""

    def __init__(self, waiting: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.waiting = waiting

    async def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if query.startswith("query BlockedByMe("):
            self.calls.append((query, variables or {}))
            return {
                "issue": {
                    "id": "issue-ENG-9",
                    "relations": {"nodes": [{"type": "blocks", "relatedIssue": self.waiting}]},
                }
            }
        return await super().execute(query, variables)


GRANDPARENT_CHAIN = {
    "id": "issue-ENG-2",
    "identifier": "ENG-2",
    "delegate": None,
    "parent": {"id": "issue-ENG-1", "identifier": "ENG-1", "delegate": {"id": "app-1"}, "parent": None},
}


async def test_blocker_done_wakes_the_delegated_ancestor_of_an_undelegated_waiter(boards):
    wakes: list[tuple[str, str, str]] = []
    waiting = _waiting_issue("ENG-5", "completed", GRANDPARENT_CHAIN)
    sync = KanbanSync(
        WaiterClient(waiting),
        wake=lambda i, k, t: wakes.append((i, k, t)),
        app_user_id="app-1",
    )
    payload = issue_payload("ENG-9", was_state="s-working", now_state="s-done")
    payload["data"]["id"] = "issue-ENG-9"

    await sync.handle_issue_webhook(payload)

    assert wakes == [("issue-ENG-1", "ENG-1", "[sync] ENG-5 unblocked.")]


async def test_waiter_failure_never_fails_the_webhook(boards):
    class Boom(WaiterClient):
        async def execute(self, query, variables=None):
            if query.startswith("query BlockedByMe("):
                raise RuntimeError("linear down")
            return await super().execute(query, variables)

    sync = KanbanSync(Boom({}), wake=lambda *a: None, app_user_id="app-1")
    payload = issue_payload("ENG-9", was_state="s-working", now_state="s-done")
    payload["data"]["id"] = "issue-ENG-9"

    line = await sync.handle_issue_webhook(payload)

    assert line.endswith("archived 0")


async def test_no_app_user_id_means_no_waiters(boards):
    sync = KanbanSync(FakeClient(), wake=lambda *a: None)

    assert sync._waiters is None


async def test_start_adds_the_waiter_loop_only_with_an_app_user_id(boards, idle_watch, tmp_path):
    boards.cursors[BOARD] = 0
    plain = KanbanSync(FakeClient(), boards_root=tmp_path / "b", reconcile_seconds=60, wake=lambda *a: None)
    waiter = KanbanSync(
        FakeClient(),
        boards_root=tmp_path / "b",
        reconcile_seconds=60,
        wake=lambda *a: None,
        app_user_id="app-1",
        waiter_seconds=3600,
    )
    all_tasks: list[asyncio.Task[None]] = []
    try:
        plain_tasks = plain.start()
        waiter_tasks = waiter.start()
        all_tasks = [*plain_tasks, *waiter_tasks]
        assert len(plain_tasks) == 2
        assert len(waiter_tasks) == 3
    finally:
        for task in all_tasks:
            task.cancel()
        await asyncio.gather(
            *all_tasks, *plain._followers.values(), *waiter._followers.values(), return_exceptions=True
        )


async def test_has_open_card_looks_at_every_board(cli):
    cli.slugs = ["a", "b"]
    cli.cards_by[("b", "ENG-5")] = [card("t1", "ready")]
    sync = KanbanSync(FakeClient())

    assert await sync._has_open_card("ENG-5") is True
    assert await sync._has_open_card("ENG-6") is False
