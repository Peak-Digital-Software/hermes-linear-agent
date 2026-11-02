"""Keep the kanban boards and their Linear issues in step.

Three paths, ported from peak-agent's bridge:

* an Issue webhook whose state move cancels/completes the issue archives that
  issue's open cards on every board, and wakes a parent session when a child
  reaches Done;
* a follower per board turns kanban card events into the issue's Linear status
  and rolls the parent issue up from its children;
* a reconcile pass re-derives every active issue's status from its whole card
  set, so a missed event heals on the next pass.

Board discovery and the per-board followers are event driven: ``watchfiles``
watches the boards root and each board's database directory, so an idle
gateway does no board reads at all. Every Linear read and write goes through
``LinearClient.execute`` with the raw GraphQL queries below; the CLI plane is
reached through ``kanban_cli`` (so tests can monkeypatch its module
attributes).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import watchfiles

from . import kanban_cli
from .kanban_rules import ACTED, CLOSED_TYPES, Card, is_open, issue_status, rollup, transition

logger = logging.getLogger(__name__)

#: watchfiles reports a batch once the filesystem has been quiet this long.
WATCH_DEBOUNCE_MS = 100
#: watchfiles re-checks the inotify queue this often while it waits.
WATCH_STEP_MS = 20

#: How often a full reconcile pass runs.
RECONCILE_SECONDS = 300.0

#: The filesystem watcher; tests monkeypatch it to drive the loops by hand.
_awatch = watchfiles.awatch

#: A card's tenant is a Linear issue key, or it is not ours.
ISSUE_KEY = re.compile(r"^[A-Z]+-\d+$")

WORKFLOW_STATE = """query WorkflowState($id: String!) {
  workflowState(id: $id) { id name type }
}"""

ISSUE = """query Issue($id: String!) {
  issue(id: $id) {
    id
    identifier
    state { id name type }
    team { id key }
    parent { id identifier }
    children { nodes { id identifier state { name } } }
  }
}"""

TEAM_STATES = """query TeamStates($teamId: String!) {
  team(id: $teamId) {
    states(first: 250) { nodes { id name type } }
  }
}"""

ISSUE_ATTACHMENTS = """query IssueAttachments($id: String!) {
  issue(id: $id) {
    attachments(first: 10) { nodes { url sourceType metadata } }
  }
}"""

ACTIVE_ISSUES = """query ActiveIssues($team: String!, $after: String) {
  issues(
    first: 50
    after: $after
    filter: {
      team: { key: { eq: $team } }
      state: { type: { in: ["backlog", "unstarted", "started"] } }
    }
  ) {
    nodes {
      id
      identifier
      title
      description
      url
      state { id name type }
      team { id key }
      parent { id identifier }
      children { nodes { id identifier state { name } } }
      attachments(first: 10) { nodes { url sourceType metadata } }
    }
    pageInfo { hasNextPage endCursor }
  }
}"""

ISSUE_UPDATE = """mutation IssueUpdate($id: String!, $stateId: String!) {
  issueUpdate(id: $id, input: { stateId: $stateId }) { success }
}"""


def _err_line(error: object) -> str:
    """One safe line for the journal, like the bridge's ``errLine``."""
    text = str(getattr(error, "stderr", None) or getattr(error, "message", None) or error)
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("Command failed:"):
            return line[:200]
    return "unknown error"


def _message(error: object) -> str:
    """The first non-empty line of an exception message."""
    for line in str(error).splitlines():
        line = line.strip()
        if line:
            return line
    return "unknown error"


def _routed(payload: dict[str, Any], flow: str) -> str:
    """One line per routing decision, the same shape the TS bridge logged."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else None
    key = data.get("identifier") if data else None
    return f"route {key or '-'}: {flow}"


def _nodes(value: object) -> list[dict[str, Any]]:
    """The ``nodes`` list of a GraphQL connection, tolerating an empty answer."""
    if not isinstance(value, dict):
        return []
    nodes = value.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, dict)]


def _db_filter(db_path: str) -> Callable[[Any, str], bool]:
    """Keep only this board's database and its write-ahead log."""
    name = os.path.basename(db_path)

    def matches(_change: Any, path: str) -> bool:
        found = os.path.basename(path)
        return found == name or found == f"{name}-wal"

    return matches


class KanbanSync:
    """The webhook, follower and reconcile loops for one Linear workspace."""

    def __init__(
        self,
        client: Any,
        team_key: str = "ENG",
        *,
        wake: Callable[[str, str, str], None] | None = None,
        boards_root: Path | None = None,
        reconcile_seconds: float = RECONCILE_SECONDS,
    ) -> None:
        self._client = client
        self.team_key = team_key
        self.reconcile_seconds = reconcile_seconds
        self._wake = wake
        self.boards_root = Path(boards_root) if boards_root is not None else Path.home() / ".hermes/kanban/boards"

        # state id -> workflow state type, resolved once per process
        self._state_types: dict[str, str | None] = {}
        # team id -> {state name: state id}, resolved once per process
        self._team_states: dict[str, dict[str, str]] = {}

        self._followers: dict[str, asyncio.Task[None]] = {}
        self._discovery_task: asyncio.Task[None] | None = None
        self._reconcile_task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> list[asyncio.Task[None]]:
        """Start board discovery and the reconcile loop; one task each."""
        self._discovery_task = asyncio.create_task(self._discover_loop())
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        return [self._discovery_task, self._reconcile_task]

    async def _discover_loop(self) -> None:
        """Follow every board, re-listing them whenever the boards root changes."""
        try:
            try:
                self.boards_root.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                logger.warning("boards root %s is unavailable: %s", self.boards_root, _err_line(error))
                return
            try:
                await self._poll_boards()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - one line, then keep watching
                logger.warning("boards() failed: %s", _err_line(error))
            try:
                async for _changes in _awatch(
                    self.boards_root,
                    recursive=False,
                    debounce=WATCH_DEBOUNCE_MS,
                    step=WATCH_STEP_MS,
                ):
                    try:
                        await self._poll_boards()
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:  # noqa: BLE001 - one line, then keep watching
                        logger.warning("boards() failed: %s", _err_line(error))
            except OSError as error:
                # The boards root itself went away; there is nothing left to watch.
                logger.warning("boards root %s: %s", self.boards_root, _err_line(error))
        finally:
            # The adapter only tracks start()'s tasks; followers live and die with discovery.
            for follower in self._followers.values():
                follower.cancel()

    async def _poll_boards(self) -> list[str]:
        """Start a follower per new board and cancel the ones that went away."""
        rows = await kanban_cli.boards()
        wanted: dict[str, str | None] = {}
        for row in rows:
            slug = row.get("slug")
            if not slug or row.get("archived"):
                continue
            wanted[slug] = row.get("db_path")

        for slug, follower in list(self._followers.items()):
            if slug in wanted and not follower.done():
                continue
            del self._followers[slug]
            follower.cancel()

        fresh: list[str] = []
        for slug, db_path in wanted.items():
            if slug in self._followers:
                continue
            self._followers[slug] = asyncio.create_task(self._follow_forever(slug, db_path))
            fresh.append(slug)
        return fresh

    # -- card events -> issue status ---------------------------------------

    async def _follow_forever(self, board: str, db_path: str | None) -> None:
        """One board's follower: seed the cursor, read, then react to changes."""
        if not db_path:
            return
        try:
            cursor = await self._seed_cursor(board, db_path)
            await self._follow(board, db_path, cursor)
        except asyncio.CancelledError:
            raise
        except FileNotFoundError:
            # The board database or its directory vanished: nothing left to watch.
            return
        except Exception as error:  # noqa: BLE001 - one line, then this follower stops
            logger.error("error: watch %s: %s", board, _err_line(error))

    async def _seed_cursor(self, board: str, db_path: str) -> int:
        """Resume from the saved cursor, or seed it from the newest event id."""
        cursor = kanban_cli.load_cursor(board, db_path)
        if cursor is not None:
            return cursor
        cursor = await asyncio.to_thread(kanban_cli.max_event_id, db_path)
        kanban_cli.save_cursor(board, db_path, cursor)
        return cursor

    async def _follow(self, board: str, db_path: str, cursor: int) -> int | None:
        """Follow one board's events until the db goes away or the watch ends."""
        folder = os.path.dirname(db_path) or "."
        latest = await self._drain(board, db_path, cursor)
        if latest is None:
            return None
        async for _changes in _awatch(
            folder,
            watch_filter=_db_filter(db_path),
            debounce=WATCH_DEBOUNCE_MS,
            step=WATCH_STEP_MS,
        ):
            latest = await self._drain(board, db_path, latest)
            if latest is None:
                return None
        return latest

    async def _drain(self, board: str, db_path: str, cursor: int) -> int | None:
        """Handle every event after ``cursor``; ``None`` when the db is gone."""
        while True:
            try:
                rows = await asyncio.to_thread(kanban_cli.read_events, db_path, cursor)
            except (FileNotFoundError, sqlite3.OperationalError) as error:
                if isinstance(error, FileNotFoundError) or not os.path.exists(db_path):
                    return None
                raise
            if not rows:
                return cursor
            for row in rows:
                try:
                    await self._handle_event(board, row)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - one bad event, keep going
                    logger.error("error: watch %s %s: %s", board, row.get("task_id"), _err_line(error))
                cursor = max(cursor, int(row.get("id", cursor)))
                kanban_cli.save_cursor(board, db_path, cursor)

    async def _handle_event(self, board: str, row: dict[str, Any]) -> None:
        kind = row.get("kind")
        if kind not in ACTED:
            return
        task_id = row.get("task_id")
        shown = await kanban_cli.show(board, task_id)
        issue_key = (shown or {}).get("tenant")
        if not issue_key or not ISSUE_KEY.match(str(issue_key)):
            return

        cards = await kanban_cli.cards(board, issue_key)
        status = issue_status(cards)
        if status == "Done":
            try:
                status = issue_status(cards, await self._attachments(issue_key))
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - hold Done on the safe side
                logger.warning(
                    "warn: watch %s %s: attachments unavailable, holding %s in Ready to Merge: %s",
                    board,
                    task_id,
                    issue_key,
                    _err_line(error),
                )
                status = "Ready to Merge"

        issue = await self._issue(issue_key)
        current = issue.get("state") or {}
        if current.get("type") in CLOSED_TYPES or status is None or status == current.get("name"):
            return
        logger.info("%s %s %s -> %s status %s", board, task_id, kind, issue_key, status)
        await self._set_status(issue_key, status)
        await self._rollup_parent(issue, status)

    async def _rollup_parent(self, issue: dict[str, Any], status: str) -> None:
        """Move the issue's parent to the rollup of its children, when it changes."""
        parent = issue.get("parent") or {}
        parent_id = parent.get("id")
        if not parent_id:
            return
        node = await self._issue(str(parent_id))
        if not node:
            return

        states: list[str] = []
        for child in _nodes(node.get("children")):
            if child.get("id") == issue.get("id"):
                # The write just made may not be visible in this read yet.
                states.append(status)
            else:
                states.append(str((child.get("state") or {}).get("name") or ""))

        rolled = rollup(states)
        if rolled is None:
            return
        current = node.get("state") or {}
        if rolled == current.get("name"):
            return
        await self._set_status(str(parent_id), rolled)
        logger.info("rollup %s: %s -> %s", parent.get("identifier") or parent_id, current.get("name"), rolled)

    # -- reconcile ---------------------------------------------------------

    async def _reconcile_loop(self) -> None:
        while True:
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - a failed pass must not kill the timer
                logger.error("reconcile failed: %s", _message(error))
            await asyncio.sleep(self.reconcile_seconds)

    async def reconcile(self) -> int:
        """One pass: mirror every active issue's card-set status; returns writes."""
        rows = await kanban_cli.boards()
        issues = await self._active_issues()
        states_by_team: dict[str, dict[str, str]] = {}
        changes = 0

        for issue in issues:
            key = issue.get("identifier")
            if not key:
                continue
            children = issue.get("children") or []
            if children:
                # A parent takes its children's rollup, never its own cards.
                status = rollup([str((child.get("state") or {}).get("name") or "") for child in children])
            else:
                cards: list[Card] = []
                for row in rows:
                    slug = row.get("slug")
                    if slug:
                        cards.extend(await kanban_cli.cards(slug, key))
                if not cards:
                    continue
                status = issue_status(cards, issue.get("attachments") or [])

            state = issue.get("state") or {}
            if status is None or status == state.get("name"):
                continue

            team_id = (issue.get("team") or {}).get("id")
            if not team_id:
                continue
            if team_id not in states_by_team:
                states_by_team[team_id] = await self._team_state_ids(team_id)
            state_id = states_by_team[team_id].get(status)
            if not state_id:
                continue  # no such state on the team: nothing safe to write

            issue_id = issue.get("id")
            if not issue_id:
                continue
            await self._update_issue(issue_id, state_id)
            logger.info("reconcile %s: %s -> %s", key, state.get("name"), status)
            changes += 1

        return changes

    # -- Issue webhook -> archive cards, wake a parent ----------------------

    async def handle_issue_webhook(self, payload: dict[str, Any]) -> str:
        """Route one verified webhook payload; returns the route result line."""
        payload_type = payload.get("type")
        action = payload.get("action") or ""
        if payload_type != "Issue" or action not in ("update", "create"):
            return _routed(payload, f"ignored: unhandled {payload_type or '-'}/{action or '-'}")

        data = payload.get("data")
        if not isinstance(data, dict):
            return _routed(payload, "ignored: no card action")

        updated = payload.get("updatedFrom")
        was_state = updated.get("stateId") if isinstance(updated, dict) else None
        now_state = data.get("stateId")
        was_type = await self._state_type(was_state) if was_state else None
        now_type = await self._state_type(now_state) if now_state else None

        card_action = transition(action, was_type, now_type)
        if card_action == "none":
            return _routed(payload, "ignored: no card action")

        key = data.get("identifier") or ""
        if now_type == "completed":
            await self._wake_parent(data, key)

        try:
            archived = await self._close_cards(key)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - log one line, never raise at the webhook
            logger.error("error: %s %s: %s", card_action, key, _err_line(error))
            return f"error: {card_action} {key}: {_err_line(error)}"
        return _routed(payload, f"{card_action}: archived {archived}")

    async def _wake_parent(self, data: dict[str, Any], key: str) -> None:
        """Wake the parent session when a sub-issue reaches Done."""
        parent_id = data.get("parentId")
        if not parent_id or self._wake is None:
            return
        try:
            found = data.get("parent")
            parent = found if isinstance(found, dict) else {}
            parent_key = str(parent.get("identifier") or "")
            if not parent_key:
                parent_key = str((await self._issue(str(parent_id))).get("identifier") or "")
            self._wake(str(parent_id), parent_key, f"[sync] {key} is Done.")
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - a wake must not fail the webhook
            logger.warning("warn: wake %s: %s", parent_id, _err_line(error))

    async def _close_cards(self, issue_key: str) -> int:
        """Archive every open card of the issue on every board (3 passes max)."""
        archived = 0
        for row in await kanban_cli.boards():
            board = row.get("slug")
            if not board:
                continue
            for _pass in range(3):
                open_ids = [card.id for card in await kanban_cli.cards(board, issue_key) if is_open(card)]
                if not open_ids:
                    break
                await kanban_cli.archive(board, open_ids)
                archived += len(open_ids)
        return archived

    # -- Linear access -----------------------------------------------------

    async def _execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        data = await self._client.execute(query, variables or {})
        return data if isinstance(data, dict) else {}

    async def _state_type(self, state_id: str) -> str | None:
        if state_id not in self._state_types:
            data = await self._execute(WORKFLOW_STATE, {"id": state_id})
            node = data.get("workflowState") or {}
            self._state_types[state_id] = node.get("type")
        return self._state_types[state_id]

    async def _issue(self, issue_key: str) -> dict[str, Any]:
        data = await self._execute(ISSUE, {"id": issue_key})
        node = data.get("issue")
        return node if isinstance(node, dict) else {}

    async def _attachments(self, issue_key: str) -> list[dict[str, Any]]:
        issue = await self._issue(issue_key)
        issue_id = issue.get("id")
        if not issue_id:
            return []
        data = await self._execute(ISSUE_ATTACHMENTS, {"id": issue_id})
        issue_data = data.get("issue") or {}
        attachments = issue_data.get("attachments") or {}
        nodes = attachments.get("nodes") or []
        return [node for node in nodes if isinstance(node, dict)]

    async def _team_state_ids(self, team_id: str) -> dict[str, str]:
        if team_id not in self._team_states:
            data = await self._execute(TEAM_STATES, {"teamId": team_id})
            team = data.get("team") or {}
            states = team.get("states") or {}
            by_name: dict[str, str] = {}
            for node in states.get("nodes") or []:
                if isinstance(node, dict) and node.get("id") and node.get("name"):
                    by_name[node["name"]] = node["id"]
            self._team_states[team_id] = by_name
        return self._team_states[team_id]

    async def _set_status(self, issue_key: str, name: str) -> None:
        issue = await self._issue(issue_key)
        team_id = (issue.get("team") or {}).get("id")
        if not team_id:
            return
        state_id = (await self._team_state_ids(team_id)).get(name)
        # A team without that state is not an error: the status stays as it is.
        if not state_id:
            return
        issue_id = issue.get("id")
        if issue_id:
            await self._update_issue(issue_id, state_id)

    async def _update_issue(self, issue_id: str, state_id: str) -> None:
        data = await self._execute(ISSUE_UPDATE, {"id": issue_id, "stateId": state_id})
        result = data.get("issueUpdate") or {}
        if not result.get("success"):
            raise RuntimeError("issueUpdate failed")

    async def _active_issues(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            data = await self._execute(ACTIVE_ISSUES, {"team": self.team_key, "after": after})
            page = data.get("issues") or {}
            for node in page.get("nodes") or []:
                normalized = _normalize_issue(node)
                if normalized is not None:
                    found.append(normalized)
            info = page.get("pageInfo") or {}
            if not info.get("hasNextPage") or not info.get("endCursor"):
                return found
            after = info.get("endCursor")


def _normalize_issue(node: object) -> dict[str, Any] | None:
    """One ActiveIssues node, flattened the way reconcile consumes it."""
    if not isinstance(node, dict) or not node.get("id") or not node.get("identifier"):
        return None
    attachments = []
    for link in (node.get("attachments") or {}).get("nodes") or []:
        if isinstance(link, dict) and isinstance(link.get("url"), str):
            attachments.append(
                {
                    "url": link["url"],
                    "sourceType": link.get("sourceType"),
                    "metadata": link.get("metadata"),
                }
            )
    children = []
    for child in _nodes(node.get("children")):
        if not child.get("id"):
            continue
        children.append(
            {
                "id": child["id"],
                "identifier": child.get("identifier") or "",
                "state": {"name": (child.get("state") or {}).get("name") or ""},
            }
        )
    parent = node.get("parent") if isinstance(node.get("parent"), dict) else {}
    state = node.get("state") or {}
    team = node.get("team") or {}
    return {
        "id": node["id"],
        "identifier": node["identifier"],
        "title": node.get("title") or "",
        "description": node.get("description"),
        "url": node.get("url") or "",
        "state": {
            "id": state.get("id") or "",
            "name": state.get("name") or "",
            "type": state.get("type") or "",
        },
        "team": {"id": team.get("id") or "", "key": team.get("key") or ""},
        "parent": {"id": parent.get("id") or "", "identifier": parent.get("identifier") or ""} if parent else None,
        "children": children,
        "attachments": attachments,
    }
