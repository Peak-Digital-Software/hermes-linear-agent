"""Tests for kanban_cli: hermes argv, card parsing, event reads, cursor state."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_linear_agent import kanban_cli
from hermes_linear_agent.kanban_rules import Card

FIXTURES = Path(__file__).parent / "fixtures" / "kanban_cli"


class FakeProc:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout.encode()
        self.stderr = stderr.encode()
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr


class FakeExec:
    """Records argv/kwargs and returns queued answers (or one fixed answer)."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(list(args[1:]))  # first arg is "hermes"
        self.kwargs.append(kwargs)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, FakeProc):
            return answer
        return FakeProc(answer)


@pytest.fixture
def fake_exec(monkeypatch):
    def install(*answers: str | FakeProc) -> FakeExec:
        fake = FakeExec(*answers)
        monkeypatch.setattr(kanban_cli.asyncio, "create_subprocess_exec", fake)
        return fake

    return install


async def test_boards_uses_exact_argv_and_returns_rows(fake_exec):
    sample = (FIXTURES / "boards.json").read_text()
    fake = fake_exec(sample)

    rows = await kanban_cli.boards()

    assert fake.calls == [["kanban", "boards", "list", "--json"]]
    assert [row["slug"] for row in rows] == ["default", "hermes-linear-agent"]
    assert rows[1]["db_path"] == "/home/db/.hermes/kanban.db"


async def test_subprocess_gets_devnull_stdin_and_pipes(fake_exec):
    fake = fake_exec("[]")

    await kanban_cli.boards()

    kwargs = fake.kwargs[0]
    assert kwargs["stdin"] is asyncio.subprocess.DEVNULL
    assert kwargs["stdout"] is asyncio.subprocess.PIPE
    assert kwargs["stderr"] is asyncio.subprocess.PIPE


async def test_cards_lists_archived_cards_and_maps_four_fields(fake_exec):
    sample = (FIXTURES / "list-eng2274.json").read_text()
    fake = fake_exec(sample)

    rows = await kanban_cli.cards("hermes-linear-agent", "ENG-2274")

    assert fake.calls == [
        ["kanban", "--board", "hermes-linear-agent", "list", "--tenant", "ENG-2274", "--archived", "--json"]
    ]
    assert len(rows) == 4
    assert rows[0] == Card(id="t_ffda41d3", status="running", assignee="coder", completed_at=None)


async def test_cards_maps_missing_assignee_and_completed_at_to_none(fake_exec):
    fake_exec(json.dumps([{"id": "t_x", "status": "archived", "completed_at": None}]))

    assert await kanban_cli.cards("b", "ENG-2") == [Card(id="t_x", status="archived", assignee=None, completed_at=None)]


async def test_show_reports_id_tenant_and_completion(fake_exec):
    fake_exec((FIXTURES / "show.json").read_text())

    assert await kanban_cli.show("hermes-linear-agent", "t_ffda41d3") == {
        "id": "t_ffda41d3",
        "tenant": "ENG-2274",
        "completed": False,
    }


async def test_show_treats_completed_at_as_done(fake_exec):
    fake_exec(json.dumps({"task": {"id": "t_a", "tenant": "ENG-1", "completed_at": 1793308374}, "parents": []}))

    assert await kanban_cli.show("b", "t_a") == {"id": "t_a", "tenant": "ENG-1", "completed": True}


async def test_archive_makes_one_call_for_every_card(fake_exec):
    fake = fake_exec("")

    await kanban_cli.archive("peak-agent-tools", ["t_a", "t_b"])

    assert fake.calls == [["kanban", "--board", "peak-agent-tools", "archive", "t_a", "t_b"]]


async def test_archive_with_no_ids_makes_no_call(fake_exec):
    fake = fake_exec("")

    await kanban_cli.archive("peak-agent-tools", [])

    assert fake.calls == []


async def test_nonzero_exit_raises_with_stderr(fake_exec):
    fake_exec(FakeProc("", stderr="board not found\n", returncode=1))

    with pytest.raises(RuntimeError, match="board not found"):
        await kanban_cli.boards()


# --- event reads ---------------------------------------------------------


def board_db(tmp_path: Path) -> Path:
    path = tmp_path / "kanban.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, assignee TEXT, "
        "status TEXT NOT NULL, completed_at INTEGER, tenant TEXT, idempotency_key TEXT);"
        "CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, "
        "run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL);"
    )
    db.commit()
    db.close()
    return path


def add_task(path: Path, task_id: str, assignee: str | None, tenant: str | None) -> None:
    db = sqlite3.connect(path)
    db.execute(
        "INSERT INTO tasks (id, title, assignee, status, tenant) VALUES (?, ?, ?, 'todo', ?)",
        (task_id, f"Card {task_id}", assignee, tenant),
    )
    db.commit()
    db.close()


def add_event(path: Path, task_id: str, kind: str, payload=None, raw: str | None = None) -> int:
    db = sqlite3.connect(path)
    text = raw if raw is not None else json.dumps({} if payload is None else payload)
    cur = db.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, 1, ?, ?, 1793300000)",
        (task_id, kind, text),
    )
    db.commit()
    event_id = int(cur.lastrowid)
    db.close()
    return event_id


def test_read_events_joins_assignee_tenant_and_decodes_payload(tmp_path):
    path = board_db(tmp_path)
    add_task(path, "t_a", "coder", "ENG-1")
    add_event(path, "t_a", "claimed", {"lock": "omarchy:1", "run_id": 2})
    add_event(path, "t_a", "blocked", raw="not json")

    rows = kanban_cli.read_events(str(path), 0)

    assert rows == [
        {
            "id": 1,
            "task_id": "t_a",
            "kind": "claimed",
            "payload": {"lock": "omarchy:1", "run_id": 2},
            "assignee": "coder",
            "tenant": "ENG-1",
        },
        {"id": 2, "task_id": "t_a", "kind": "blocked", "payload": "not json", "assignee": "coder", "tenant": "ENG-1"},
    ]
    assert [row["id"] for row in kanban_cli.read_events(str(path), 1)] == [2]
    assert [row["id"] for row in kanban_cli.read_events(str(path), 0, 1)] == [1]


def test_read_events_pages_200_rows_in_id_order(tmp_path):
    path = board_db(tmp_path)
    add_task(path, "t_a", "coder", "ENG-1")
    for index in range(205):
        add_event(path, "t_a", "created", {"index": index})

    first = kanban_cli.read_events(str(path), 0)
    second = kanban_cli.read_events(str(path), first[-1]["id"])

    assert len(first) == 200
    assert [row["id"] for row in first] == list(range(1, 201))
    assert len(second) == 5
    assert [row["id"] for row in second] == list(range(201, 206))


def test_max_event_id_is_zero_on_empty_board_and_newest_otherwise(tmp_path):
    path = board_db(tmp_path)
    assert kanban_cli.max_event_id(str(path)) == 0
    add_event(path, "t_a", "created")
    newest = add_event(path, "t_a", "claimed")
    assert kanban_cli.max_event_id(str(path)) == newest


def test_read_events_opens_sqlite_read_only(tmp_path):
    path = board_db(tmp_path)

    conn = kanban_cli._open(str(path))
    with pytest.raises(sqlite3.OperationalError):
        # a read-only connection cannot create a table
        conn.execute("CREATE TABLE nope (id INTEGER)")
    conn.close()


# --- cursor state --------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAK_AGENT_STATE_DIR", str(tmp_path))
    return tmp_path


def test_cursor_is_persisted_per_board_and_other_boards_are_kept(state_dir):
    assert kanban_cli.load_cursor("alpha") is None
    kanban_cli.save_cursor("alpha", 5)
    kanban_cli.save_cursor("beta", 9)

    assert kanban_cli.load_cursor("alpha") == 5
    assert kanban_cli.load_cursor("beta") == 9
    assert json.loads((state_dir / "watch-cursor.json").read_text()) == {"alpha": 5, "beta": 9}
    assert not list(state_dir.glob("*.tmp")), "the write is a rename, no temp file is left behind"


def test_corrupt_cursor_file_is_ignored_not_fatal(state_dir):
    (state_dir / "watch-cursor.json").write_text("not json")

    assert kanban_cli.load_cursor("alpha") is None


def test_save_cursor_recovers_from_a_corrupt_file(state_dir):
    (state_dir / "watch-cursor.json").write_text("not json")

    kanban_cli.save_cursor("alpha", 3)

    assert json.loads((state_dir / "watch-cursor.json").read_text()) == {"alpha": 3}


def test_cursor_defaults_to_the_linear_agent_state_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("PEAK_AGENT_STATE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert kanban_cli._cursor_file() == tmp_path / ".local" / "state" / "linear-agent" / "watch-cursor.json"


# --- follower ------------------------------------------------------------


async def test_follow_cold_start_seeds_at_max_id_and_handles_only_later_rows(tmp_path, state_dir):
    path = board_db(tmp_path)
    add_task(path, "t_a", "coder", "ENG-1")
    for _ in range(3):
        add_event(path, "t_a", "created")

    seen: list[int] = []
    stop = asyncio.Event()

    async def on_event(row):
        seen.append(row["id"])
        stop.set()

    task = asyncio.create_task(
        kanban_cli.follow("peak-agent-tools", on_event, db_path=str(path), interval=0.01, stop=stop)
    )
    await asyncio.sleep(0.05)
    assert seen == [], "history is not replayed on a cold start"

    add_event(path, "t_a", "completed")
    await asyncio.wait_for(task, timeout=2)

    assert seen == [4]
    assert kanban_cli.load_cursor("peak-agent-tools") == 4


async def test_follow_resumes_from_the_saved_cursor(tmp_path, state_dir):
    path = board_db(tmp_path)
    add_task(path, "t_a", "coder", "ENG-1")
    add_event(path, "t_a", "created")
    add_event(path, "t_a", "claimed")
    kanban_cli.save_cursor("peak-agent-tools", 0)

    seen: list[int] = []
    stop = asyncio.Event()

    async def on_event(row):
        seen.append(row["id"])
        if len(seen) == 2:
            stop.set()

    await asyncio.wait_for(
        kanban_cli.follow("peak-agent-tools", on_event, db_path=str(path), interval=0.01, stop=stop), timeout=2
    )

    assert seen == [1, 2]
    assert kanban_cli.load_cursor("peak-agent-tools") == 2


async def test_follow_logs_an_event_failure_and_still_advances(tmp_path, state_dir, caplog):
    path = board_db(tmp_path)
    add_task(path, "t_a", "coder", "ENG-1")
    add_event(path, "t_a", "created")
    add_event(path, "t_a", "claimed")
    kanban_cli.save_cursor("peak-agent-tools", 0)

    seen: list[int] = []
    stop = asyncio.Event()

    async def on_event(row):
        seen.append(row["id"])
        if row["id"] == 1:
            raise RuntimeError("Linear is down")
        stop.set()

    with caplog.at_level("ERROR"):
        await asyncio.wait_for(
            kanban_cli.follow("peak-agent-tools", on_event, db_path=str(path), interval=0.01, stop=stop), timeout=2
        )

    assert seen == [1, 2]
    assert kanban_cli.load_cursor("peak-agent-tools") == 2
    assert "error: watch peak-agent-tools t_a: Linear is down" in caplog.text


async def test_follow_returns_immediately_when_stop_is_already_set(tmp_path, state_dir):
    path = board_db(tmp_path)
    stop = asyncio.Event()
    stop.set()

    async def on_event(row):  # pragma: no cover - must not run
        raise AssertionError("must not handle")

    await kanban_cli.follow("peak-agent-tools", on_event, db_path=str(path), interval=0.01, stop=stop)
