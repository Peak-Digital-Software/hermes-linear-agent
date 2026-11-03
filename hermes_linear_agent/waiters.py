"""Wake the session that is waiting when a blocker finishes (ENG-2343).

Linear sends ``issueUnblocked`` only for issues involving the app, so an
undelegated sub-issue whose blocker goes Done tells nobody. The kanban sync
sees every issue update and asks this module who was waiting: the nearest
delegated ancestor of each issue the closed blocker blocked.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

CLOSED = frozenset({"completed", "canceled"})
PARENT_DEPTH = 4  # parent chain levels fetched


def _parent_chain(depth: int) -> str:
    inner = ""
    for _ in range(depth):
        inner = f"parent {{ id identifier delegate {{ id }} {inner} }}"
    return inner


_NODE = (
    "id identifier state { type } delegate { id } "
    "inverseRelations(first: 50) { nodes { type issue { id identifier state { type } } } } "
    + _parent_chain(PARENT_DEPTH)
)

BLOCKED_BY_ME = (
    "query BlockedByMe($id: String!) {\n"
    "  issue(id: $id) {\n"
    "    id\n"
    "    relations(first: 50) { nodes { type relatedIssue { " + _NODE + " } } }\n"
    "  }\n"
    "}"
)

WAITING = (
    "query Waiting($team: String!, $after: String) {\n"
    "  issues(\n"
    "    first: 50\n"
    "    after: $after\n"
    "    filter: {\n"
    "      team: { key: { eq: $team } }\n"
    '      state: { type: { in: ["backlog", "unstarted", "started"] } }\n'
    "      hasBlockedByRelations: { eq: true }\n"
    "    }\n"
    "  ) {\n"
    "    nodes { " + _NODE + " }\n"
    "    pageInfo { hasNextPage endCursor }\n"
    "  }\n"
    "}"
)

NEAREST = "query Nearest($id: String!) {\n  issue(id: $id) { " + _NODE + " }\n}"


def blockers(issue: dict) -> list[dict]:
    """The issue's blocker nodes: inverseRelations nodes with type "blocks" -> node["issue"]."""
    nodes = ((issue or {}).get("inverseRelations") or {}).get("nodes") or []
    found = []
    for node in nodes:
        if isinstance(node, dict) and node.get("type") == "blocks" and isinstance(node.get("issue"), dict):
            found.append(node["issue"])
    return found


def _closed(issue: dict) -> bool:
    return str((issue.get("state") or {}).get("type") or "") in CLOSED


def all_blockers_closed(issue: dict) -> bool:
    """True when it has >= 1 blocker and every blocker's state.type is in CLOSED."""
    found = blockers(issue)
    return bool(found) and all(_closed(blocker) for blocker in found)


def nearest_delegated(issue: dict, app_user_id: str) -> tuple[str, str] | None:
    """(id, identifier) of the issue itself if delegate.id == app_user_id, else the first parent up the
    fetched chain whose delegate.id == app_user_id; None if none or app_user_id empty."""
    if not app_user_id:
        return None
    node: Any = issue
    while isinstance(node, dict):
        if (node.get("delegate") or {}).get("id") == app_user_id and node.get("id"):
            return str(node["id"]), str(node.get("identifier") or "")
        node = node.get("parent")
    return None


class Waiters:
    def __init__(
        self,
        client: Any,
        app_user_id: str,
        wake: Callable[[str, str, str], None],
        has_open_card: Callable[[str], Awaitable[bool]],
    ) -> None:
        self._client = client
        self._app_user_id = app_user_id
        self._wake = wake
        self._has_open_card = has_open_card
        self._woken: set[tuple[str, str, frozenset[str]]] = set()

    def _notify(self, issue: dict) -> str | None:
        """Wake the nearest delegated ancestor for one waiting issue, once per blocker set."""
        target = nearest_delegated(issue, self._app_user_id)
        if target is None:
            return None
        waiting_id = str(issue.get("id") or "")
        stamp = (target[0], waiting_id, frozenset(str(b.get("id") or "") for b in blockers(issue)))
        if stamp in self._woken:
            return None
        self._wake(target[0], target[1], f"[sync] {issue.get('identifier') or waiting_id} unblocked.")
        self._woken.add(stamp)
        return target[1]

    async def on_blocker_closed(self, blocker_id: str) -> list[str]:
        """Fetch issues the blocker blocks (BLOCKED_BY_ME query); for each one not itself closed with
        all_blockers_closed: wake nearest_delegated with text "[sync] <key> unblocked." Returns woken target keys."""
        if not self._app_user_id:
            return []
        data = await self._client.execute(BLOCKED_BY_ME, {"id": blocker_id})
        nodes = ((data.get("issue") or {}).get("relations") or {}).get("nodes") or []
        woken: list[str] = []
        for node in nodes:
            if not isinstance(node, dict) or node.get("type") != "blocks":
                continue
            issue = node.get("relatedIssue")
            if not isinstance(issue, dict):
                continue
            try:
                if _closed(issue) or not all_blockers_closed(issue):
                    continue
                key = self._notify(issue)
            except Exception as error:  # noqa: BLE001 - one failing issue must not stop the rest
                logger.error("waiter wake for %s failed: %s", issue.get("identifier"), error)
                continue
            if key:
                woken.append(key)
        return woken

    async def reconcile(self, team_keys: tuple[str, ...]) -> list[str]:
        """WAITING query per team (active states, paged): each issue with all_blockers_closed and
        not await has_open_card(identifier) -> wake nearest_delegated, same text. Returns woken keys."""
        if not self._app_user_id:
            return []
        woken: list[str] = []
        for team in team_keys:
            after: str | None = None
            while True:
                data = await self._client.execute(WAITING, {"team": team, "after": after})
                page = data.get("issues") or {}
                for issue in page.get("nodes") or []:
                    if not isinstance(issue, dict):
                        continue
                    try:
                        if _closed(issue) or not all_blockers_closed(issue):
                            continue
                        if await self._has_open_card(str(issue.get("identifier") or "")):
                            continue
                        key = self._notify(issue)
                    except Exception as error:  # noqa: BLE001 - one failing issue must not stop the rest
                        logger.error("waiter reconcile for %s failed: %s", issue.get("identifier"), error)
                        continue
                    if key:
                        woken.append(key)
                info = page.get("pageInfo") or {}
                if not info.get("hasNextPage") or not info.get("endCursor"):
                    break
                after = info["endCursor"]
        return woken

    async def nearest_for(self, issue_id: str) -> tuple[str, str] | None:
        """Fetch one issue with its parent chain (NEAREST query) and return nearest_delegated."""
        data = await self._client.execute(NEAREST, {"id": issue_id})
        issue = data.get("issue")
        if not isinstance(issue, dict):
            return None
        return nearest_delegated(issue, self._app_user_id)
