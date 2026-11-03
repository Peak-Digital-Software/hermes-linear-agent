"""Pure status rules for the kanban <-> Linear bridge.

No I/O, no Hermes imports: every function here is a total function of its
arguments, so the bridge's decisions can be tested without a board, a
database or a Linear client. The behaviour is ported from
``peak-agent-tools/peak-agent`` (``issuesync.ts``, ``watch.ts``,
``blocked.ts``, ``reconcile.ts``) — port the behaviour, not the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Linear workflow state types that mean the issue is closed (D3).
CLOSED_TYPES: frozenset[str] = frozenset({"completed", "canceled", "duplicate"})

#: Linear state types that archive the issue's cards instead of closing the issue.
_CANCEL_TYPES: frozenset[str] = frozenset({"canceled", "duplicate"})

#: The workflow state a blocked card puts its issue in.
BLOCKED_STATE = "Blocked"

#: Per-team renames of the canonical (ENG) status names; a team not listed uses them as-is.
_OPS_STATE_NAMES = {"Working": "In Progress", "Reviewing": "In Review", "Ready to Merge": "In Review"}
TEAM_STATE_NAMES: dict[str, dict[str, str]] = {"OPS": _OPS_STATE_NAMES, "ENB": _OPS_STATE_NAMES}

#: States a human sets by hand; the sync never moves an issue out of them.
HAND_STATES: dict[str, frozenset[str]] = {
    "OPS": frozenset({"Waiting on client"}),
    "ENB": frozenset({"Waiting on client"}),
}


def team_state_name(team_key: str | None, status: str) -> str:
    """The team's name for a canonical status."""
    return TEAM_STATE_NAMES.get(team_key or "", {}).get(status, status)


def hand_set(team_key: str | None, state_name: str | None) -> bool:
    """True when ``state_name`` is one a human sets by hand for this team."""
    return state_name in HAND_STATES.get(team_key or "", frozenset())


#: Card statuses that mean the card can no longer change (D3).
_CARD_CLOSED_STATUSES: frozenset[str] = frozenset({"done", "archived"})

#: Open card statuses that make an issue "Todo" (row 4 of issue_status).
_TODO_STATUSES: frozenset[str] = frozenset({"todo", "ready", "triage"})

#: Child states that mean work has started, for rollup.
_STARTED_STATES: frozenset[str] = frozenset({"Working", "Reviewing", "Ready to Merge", "In Progress", "In Review"})

#: Every kanban event this bridge acts on (watch.ts:54-68). Anything else —
#: heartbeat, worker_bound, dependency_wait, ... — never reaches Linear.
ACTED: frozenset[str] = frozenset(
    {
        "created",
        "claimed",
        "spawned",
        "unblocked",
        "changes_requested",
        "review_requested",
        "completed",
        "archived",
        "pr_acceptance",
        "blocked",
        "crashed",
        "gave_up",
        "timed_out",
    }
)

IssueAction = Literal["cancel", "close", "none"]
PrState = Literal["open", "merged", "none"]


@dataclass(frozen=True)
class Card:
    """One kanban card, as far as the status rules care."""

    id: str
    status: str
    assignee: str | None
    completed_at: int | None


def transition(action: str, was_type: str | None, type_: str | None) -> IssueAction:
    """The card action one Issue webhook asks for.

    ``was_type`` is the payload's previous state type, or ``None`` when it
    names none (a create, or an update that changed another field). Only two
    moves touch cards: a non-closed, non-Triage issue canceled/duplicated
    (archive), and an issue reaching Done (archive). Everything else — a
    create, a Triage move, a re-open — is ``"none"``: the issue's agent
    creates the cards.
    """
    # a create and an update that never changed state are not moves at all
    if action != "update" or was_type is None or was_type == type_:
        return "none"

    if was_type != "triage" and was_type not in CLOSED_TYPES and type_ is not None and type_ in _CANCEL_TYPES:
        return "cancel"

    if was_type != "completed" and type_ == "completed":
        return "close"

    return "none"


def is_open(card: Card) -> bool:
    """A card that can still change: neither done nor archived (D3)."""
    return card.status not in _CARD_CLOSED_STATUSES


def is_completed(card: Card) -> bool:
    """A card is completed once it is done or carries a completion time."""
    return card.status == "done" or card.completed_at is not None


def _is_pr_number(value: object) -> bool:
    """``typeof value === "number"`` for JSON numbers, but not for bools."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pr_gate(attachments: list[dict]) -> PrState:
    """The PR state an issue's attachments report (ENG-2239).

    A PR attachment is a ``github`` link carrying ``metadata.number``; a repo
    link or any other attachment never counts. ``"open"`` when any PR is
    neither merged nor closed (a missing status counts as open), ``"merged"``
    when PRs exist and none is open, ``"none"`` when the issue has no PR.
    """
    prs = [
        link
        for link in attachments
        if isinstance(link, dict)
        and link.get("sourceType") == "github"
        and isinstance(link.get("metadata"), dict)
        and _is_pr_number(link["metadata"].get("number"))
    ]
    if not prs:
        return "none"
    for pr in prs:
        status = pr["metadata"].get("status")
        if status != "merged" and status != "closed":
            return "open"
    return "merged"


def issue_status(cards: list[Card], attachments: list[dict] | None = None) -> str | None:
    """The issue's Linear status, read from ALL of its cards.

    ENG-2286: an eight-row precedence, first match wins.

    1. any blocked card -> ``"Blocked"``
    2. any running card -> ``"Working"``
    3. any card in review -> ``"Reviewing"``
    4. every open card is todo/ready/triage -> ``"Todo"``
    5. every card completed and a PR is open -> ``"Ready to Merge"``
    6. every card completed, PR merged or none -> ``"Done"``
    7. every card archived and none completed -> ``"Canceled"``
    8. no cards -> ``None``

    ``None`` when the issue has no card at all, so nothing is touched. An
    open card that matches none of rows 1-4 counts as active work
    (``"Working"``). The attachments gate rows 5/6: while any PR is open the
    issue sits in Ready to Merge (ENG-2239).
    """
    links = attachments if attachments is not None else []

    if not cards:
        return None
    if any(card.status == "blocked" for card in cards):
        return "Blocked"
    if any(card.status == "running" for card in cards):
        return "Working"
    if any(card.status == "review" for card in cards):
        return "Reviewing"

    open_cards = [card for card in cards if is_open(card)]
    if open_cards and all(card.status in _TODO_STATUSES for card in open_cards):
        return "Todo"

    # Rows 5/6 need every card closed with real work done. One completed
    # card is real work whatever dead duplicate was archived beside it
    # (ENG-2230, ENG-2234); only an all-archived, never-completed set is
    # Canceled (row 7).
    if not open_cards and not all(card.status == "archived" and not is_completed(card) for card in cards):
        return "Ready to Merge" if pr_gate(links) == "open" else "Done"
    if not open_cards:
        return "Canceled"

    return "Working"


def rollup(child_states: list[str]) -> str | None:
    """The parent card's status, rolled up from its child cards (ENG-2286).

    ``"Blocked"`` when any child is blocked; ``"Working"`` when any child has
    started (Working, Reviewing or Ready to Merge); ``None`` otherwise, so
    the parent is left untouched (all children done/canceled, none started,
    or no children at all — the parent session closes the parent).
    """
    if BLOCKED_STATE in child_states:
        return BLOCKED_STATE
    if any(state in _STARTED_STATES for state in child_states):
        return "Working"
    return None
