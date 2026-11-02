"""Thin async wrapper around the `hermes kanban` CLI, plus the event cursor.

Behaviour ported from peak-agent's `hermes.ts` and `cursor.ts`:

* every CLI call is `hermes <args>` with stdin closed and a 60 s timeout;
  non-zero exit raises ``RuntimeError`` carrying stderr;
* cards come from ``list --tenant KEY --archived --json`` so a canceled issue is
  still visible through its archived card;
* events are read straight from ``task_events`` with the same query Hermes'
  own watcher uses, on a read-only SQLite connection;
* the cursor is one flat ``{board: event_id}`` JSON file, written via tmp +
  rename so a crash cannot truncate it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from .kanban_rules import Card

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60.0

# Mirrors kanban_ops.py (LEFT JOIN tasks, id > cursor, ORDER BY id ASC LIMIT n).
EVENTS_SQL = """SELECT e.id, e.task_id, e.kind, e.payload, t.assignee, t.tenant
  FROM task_events e LEFT JOIN tasks t ON t.id = e.task_id
  WHERE e.id > ? ORDER BY e.id ASC LIMIT ?"""
MAX_ID_SQL = "SELECT COALESCE(MAX(id), 0) AS id FROM task_events"

STATE_FILE = "watch-cursor.json"


async def _exec(args: list[str]) -> str:
    """Run ``hermes`` with ``args``; return trimmed stdout, raise with stderr."""
    proc = await asyncio.create_subprocess_exec(
        "hermes",
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        raise RuntimeError(f"hermes {args[0] if args else ''} timed out after {TIMEOUT_SECONDS:.0f}s") from None
    if proc.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise RuntimeError(message or f"hermes {' '.join(args)} exited {proc.returncode}")
    return stdout.decode(errors="replace").strip()


async def boards() -> list[dict]:
    """Every board row as ``hermes kanban boards list --json`` reports it."""
    out = await _exec(["kanban", "boards", "list", "--json"])
    return json.loads(out)


async def db_path_for(board: str) -> str:
    """The board database path reported by ``boards list --json``."""
    rows = await boards()
    for row in rows:
        if row.get("slug") == board and row.get("db_path"):
            return str(row["db_path"])
    raise RuntimeError(f"no db_path for board {board}")


def _to_card(row: dict) -> Card:
    return Card(
        id=row["id"],
        status=row["status"],
        assignee=row.get("assignee"),
        completed_at=row.get("completed_at"),
    )


async def cards(board: str, key: str) -> list[Card]:
    """Every card of the issue, archived ones included."""
    out = await _exec(["kanban", "--board", board, "list", "--tenant", key, "--archived", "--json"])
    return [_to_card(row) for row in json.loads(out)]


async def show(board: str, task_id: str) -> dict:
    """One card as ``show --json`` reports it: id, tenant and completion."""
    out = await _exec(["kanban", "--board", board, "show", task_id, "--json"])
    data = json.loads(out)
    task = data["task"]
    return {
        "id": task["id"],
        "tenant": task.get("tenant"),
        "completed": task.get("completed_at") is not None,
    }


async def archive(board: str, ids: list[str]) -> None:
    """Archive cards; an empty list is no call at all."""
    if not ids:
        return
    await _exec(["kanban", "--board", board, "archive", *ids])


# --- event reads ---------------------------------------------------------


def _open(db_path: str) -> sqlite3.Connection:
    """Open the board DB read-only; the dispatcher may be mid-write."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _decode_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def read_events(db_path: str, after: int, limit: int = 200) -> list[dict]:
    """Events after ``after``, oldest first, payload decoded when it parses."""
    conn = _open(db_path)
    try:
        rows = conn.execute(EVENTS_SQL, (after, limit)).fetchall()
        return [
            {
                "id": int(row["id"]),
                "task_id": row["task_id"],
                "kind": row["kind"],
                "payload": _decode_payload(row["payload"]),
                "assignee": row["assignee"],
                "tenant": row["tenant"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def max_event_id(db_path: str) -> int:
    """The newest event id on the board, 0 when it has no events."""
    conn = _open(db_path)
    try:
        row = conn.execute(MAX_ID_SQL).fetchone()
        return int(row["id"] or 0) if row is not None else 0
    finally:
        conn.close()


# --- cursor state --------------------------------------------------------


def _state_dir() -> Path:
    override = os.environ.get("PEAK_AGENT_STATE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "state" / "linear-agent"


def _cursor_file() -> Path:
    return _state_dir() / STATE_FILE


def load_cursor(board: str) -> int | None:
    """The board's last handled event id, or None when nothing is saved/corrupt."""
    try:
        parsed = json.loads(_cursor_file().read_text())
    except (OSError, ValueError):
        return None
    event_id = parsed.get(board) if isinstance(parsed, dict) else None
    if isinstance(event_id, bool) or not isinstance(event_id, (int, float)):
        return None
    if isinstance(event_id, float) and not event_id.is_integer():
        return None
    return int(event_id)


def save_cursor(board: str, event_id: int) -> None:
    """Persist the board's cursor via tmp + rename; other boards are kept."""
    file = _cursor_file()
    cursors: dict[str, Any] = {}
    try:
        parsed = json.loads(file.read_text())
        if isinstance(parsed, dict):
            cursors = parsed
    except (OSError, ValueError):
        # no state yet, or corrupt: start over rather than crash the watcher
        pass
    cursors[board] = event_id
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cursors))
    tmp.replace(file)


# --- follower ------------------------------------------------------------


def _error_line(error: BaseException) -> str:
    """One safe line for the log, like watch.ts does."""
    text = str(error) or error.__class__.__name__
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("Command failed:"):
            return line[:200]
    return "unknown error"


async def _pause(interval: float, stop: asyncio.Event | None) -> None:
    if stop is None:
        await asyncio.sleep(interval)
        return
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=interval)


async def follow(
    board: str,
    on_event,
    *,
    db_path: str | None = None,
    interval: float = 1.0,
    stop: asyncio.Event | None = None,
) -> None:
    """Follow a board's events until ``stop`` is set.

    Resumes from the saved cursor; on a cold start it seeds at ``MAX(id)`` so
    history is not replayed. A failing ``on_event`` is logged and the cursor
    still advances, exactly like the TypeScript watcher.
    """
    if stop is not None and stop.is_set():
        return
    if db_path is None:
        db_path = await db_path_for(board)
    cursor = load_cursor(board)
    if cursor is None:
        cursor = max_event_id(db_path)

    while stop is None or not stop.is_set():
        rows = read_events(db_path, cursor)
        if not rows:
            await _pause(interval, stop)
            continue
        for row in rows:
            try:
                await on_event(row)
            except Exception as error:  # noqa: BLE001 - one bad event must not stop the follower
                log.error("error: watch %s %s: %s", board, row["task_id"], _error_line(error))
            cursor = max(cursor, row["id"])
            try:
                save_cursor(board, cursor)
            except Exception as error:  # noqa: BLE001 - losing the cursor is not fatal
                log.error("error: watch %s: cursor not saved: %s", board, _error_line(error))
            if stop is not None and stop.is_set():
                break
