"""Thin async wrapper around the `hermes kanban` CLI, plus the event cursor.

Behaviour ported from peak-agent's `hermes.ts` and `cursor.ts`:

* every CLI call is `hermes <args>` with stdin closed and a 60 s timeout;
  non-zero exit raises ``RuntimeError`` carrying stderr;
* cards come from ``list --tenant KEY --archived --json`` so a canceled issue is
  still visible through its archived card;
* events are read straight from ``task_events`` with the same query Hermes'
  own watcher uses, on a read-only SQLite connection;
* the cursor is one flat ``{board@inode: event_id}`` JSON file, written via
  tmp + rename so a crash cannot truncate it; a bare board slug is migrated
  on the next save.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .kanban_rules import Card

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


async def done_when(board: str, task_id: str) -> str | None:
    """The ``done_when`` metadata of the card's latest run; None on any error (ENG-2378)."""
    try:
        out = await _exec(["kanban", "--board", board, "show", task_id, "--json"])
        runs = [run for run in json.loads(out).get("runs") or [] if isinstance(run, dict)]
        if not runs:
            return None
        latest = max(enumerate(runs), key=lambda pair: (pair[1].get("ended_at") or 0, pair[0]))[1]
        value = (latest.get("metadata") or {}).get("done_when")
        return value if isinstance(value, str) else None
    except Exception:  # noqa: BLE001 - no evidence is the safe answer
        return None


async def create(
    board: str,
    title: str,
    *,
    body: str,
    assignee: str,
    tenant: str,
    idempotency_key: str,
    workspace: str,
    completion_contract: str = "local-only",
    skills: list[str] | None = None,
) -> str:
    """Create (or, same key, return) a card; returns its id.

    The body goes through a temp file and ``--body-file``; the file is removed
    after the call, also on error.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        out = await _exec(
            [
                "kanban",
                "--board",
                board,
                "create",
                title,
                "--body-file",
                path,
                "--assignee",
                assignee,
                "--tenant",
                tenant,
                "--idempotency-key",
                idempotency_key,
                "--workspace",
                workspace,
                "--completion-contract",
                completion_contract,
                *[arg for skill in skills or [] for arg in ("--skill", skill)],
                "--json",
            ]
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
    return str(json.loads(out)["id"])


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


def _cursor_key(board: str, db_path: str) -> str:
    """``board@inode``; a missing db falls back to the bare slug."""
    try:
        inode = os.stat(db_path).st_ino
    except OSError:
        return board
    return f"{board}@{inode}"


def _coerce_event_id(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def load_cursor(board: str, db_path: str) -> int | None:
    """The board's last handled event id, or None when nothing is saved/corrupt."""
    try:
        parsed = json.loads(_cursor_file().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    key = _cursor_key(board, db_path)
    if key not in parsed and key != board:
        # Migration: honour the pre-inode bare slug key when the inode key
        # has not been written yet.
        key = board
    return _coerce_event_id(parsed.get(key))


def save_cursor(board: str, db_path: str, event_id: int) -> None:
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
    key = _cursor_key(board, db_path)
    cursors[key] = event_id
    if key != board:
        # Migration: the inode key replaces the bare slug key.
        cursors.pop(board, None)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cursors))
    tmp.replace(file)
