"""Keep the kanban boards and their Linear issues in step.

Three paths, ported from peak-agent's bridge:

* an Issue webhook whose state move cancels/completes the issue archives that
  issue's open cards on every board;
* a follower per board turns kanban card events into the issue's Linear status;
* a reconcile pass re-derives every active issue's status from its whole card
  set, so a missed event heals on the next pass.

Every Linear read and write goes through ``LinearClient.execute`` with the raw
GraphQL queries below; the CLI plane is reached through ``kanban_cli`` (so tests
can monkeypatch its module attributes).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from . import kanban_cli
from .kanban_rules import ACTED, CLOSED_TYPES, Card, is_open, issue_status, transition, with_blocked

logger = logging.getLogger(__name__)

#: How often new boards are discovered.
DISCOVERY_SECONDS = 60.0
#: How often a full reconcile pass runs.
RECONCILE_SECONDS = 300.0
#: How often an idle board follower polls its event table.
POLL_SECONDS = 1.0

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


class KanbanSync:
    """The webhook, follower and reconcile loops for one Linear workspace."""

    def __init__(
        self,
        client: Any,
        team_key: str = "ENG",
        *,
        discovery_seconds: float = DISCOVERY_SECONDS,
        reconcile_seconds: float = RECONCILE_SECONDS,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._client = client
        self.team_key = team_key
        self.discovery_seconds = discovery_seconds
        self.reconcile_seconds = reconcile_seconds
        self.poll_seconds = poll_seconds

        # state id -> workflow state type, resolved once per process
        self._state_types: dict[str, str | None] = {}
        # team id -> {state name: state id}, resolved once per process
        self._team_states: dict[str, dict[str, str]] = {}

        self._started_boards: set[str] = set()
        self._followers: list[asyncio.Task[None]] = []
        self._discovery_task: asyncio.Task[None] | None = None
        self._reconcile_task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> list[asyncio.Task[None]]:
        """Start board discovery and the reconcile loop; one task each."""
        self._discovery_task = asyncio.create_task(self._discover_loop())
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        return [self._discovery_task, self._reconcile_task]

    async def _discover_loop(self) -> None:
        try:
            while True:
                try:
                    await self._poll_boards()
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - one line, then try again
                    logger.warning("boards() failed: %s", _err_line(error))
                await asyncio.sleep(self.discovery_seconds)
        finally:
            # The adapter only tracks start()'s tasks; followers live and die with discovery.
            for follower in self._followers:
                follower.cancel()

    async def _poll_boards(self) -> list[str]:
        """Start one follower per board slug this process has not seen yet."""
        rows = await kanban_cli.boards()
        fresh: list[str] = []
        for row in rows:
            slug = row.get("slug")
            if not slug or slug in self._started_boards:
                continue
            self._started_boards.add(slug)
            self._followers.append(asyncio.create_task(self._follow_forever(slug, row.get("db_path"))))
            fresh.append(slug)
        return fresh

    # -- card events -> issue status ---------------------------------------

    async def _follow_forever(self, board: str, db_path: str | None) -> None:
        while True:
            try:
                # _follow never returns, so a retry must resume from the persisted
                # cursor, not the stale one it started with.
                cursor = await self._seed_cursor(board, db_path)
                await self._follow(board, db_path, cursor)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - a follower never dies
                logger.error("error: watch %s: %s", board, _err_line(error))
                await asyncio.sleep(self.poll_seconds)

    async def _seed_cursor(self, board: str, db_path: str | None) -> int:
        """Resume from the saved cursor, or seed it from the newest event id."""
        cursor = kanban_cli.load_cursor(board)
        if cursor is not None:
            return cursor
        cursor = await asyncio.to_thread(kanban_cli.max_event_id, db_path)
        kanban_cli.save_cursor(board, cursor)
        return cursor

    async def _follow(self, board: str, db_path: str | None, cursor: int) -> int:
        """Follow one board's events until cancelled; returns the last cursor."""
        while True:
            rows = await asyncio.to_thread(kanban_cli.read_events, db_path, cursor)
            if not rows:
                await asyncio.sleep(self.poll_seconds)
                continue
            for row in rows:
                try:
                    await self._handle_event(board, row)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - one bad event, keep going
                    logger.error("error: watch %s %s: %s", board, row.get("task_id"), _err_line(error))
                cursor = max(cursor, int(row.get("id", cursor)))
                kanban_cli.save_cursor(board, cursor)

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
        base = with_blocked(cards, issue_status(cards))
        status = base
        if base == "Done":
            try:
                status = with_blocked(cards, issue_status(cards, await self._attachments(issue_key)))
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - hold Done on the safe side
                logger.warning(
                    "warn: watch %s %s: attachments unavailable, holding %s in Reviewing: %s",
                    board,
                    task_id,
                    issue_key,
                    _err_line(error),
                )
                status = with_blocked(cards, "Reviewing")

        current = await self._current_status(issue_key)
        closed = current.get("type") in CLOSED_TYPES
        if closed or status is None or status == current.get("name"):
            return
        logger.info("%s %s %s -> %s status %s", board, task_id, kind, issue_key, status)
        await self._set_status(issue_key, status)

    async def _current_status(self, issue_key: str) -> dict[str, Any]:
        issue = await self._issue(issue_key)
        state = issue.get("state") or {}
        return {"name": state.get("name"), "type": state.get("type")}

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
            cards: list[Card] = []
            for row in rows:
                slug = row.get("slug")
                if slug:
                    cards.extend(await kanban_cli.cards(slug, key))
            if not cards:
                continue

            state = issue.get("state") or {}
            status = with_blocked(cards, issue_status(cards, issue.get("attachments") or []))
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

    # -- Issue webhook -> archive cards ------------------------------------

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
        try:
            archived = await self._close_cards(key)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - log one line, never raise at the webhook
            logger.error("error: %s %s: %s", card_action, key, _err_line(error))
            return f"error: {card_action} {key}: {_err_line(error)}"
        return _routed(payload, f"{card_action}: archived {archived}")

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
        "attachments": attachments,
    }
