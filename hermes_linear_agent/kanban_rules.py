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

#: Card statuses that mean the card can no longer change (D3).
_CARD_CLOSED_STATUSES: frozenset[str] = frozenset({"done", "archived"})

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

    ``None`` when the issue has no card at all, so nothing is touched. The
    attachments gate Done: while any PR is open the issue stays in Reviewing
    (ENG-2239).
    """
    links = attachments if attachments is not None else []

    def done() -> str:
        return "Reviewing" if pr_gate(links) == "open" else "Done"

    if not cards:
        return None
    if all(is_completed(card) for card in cards):
        return done()

    remaining = [card for card in cards if is_open(card)]
    if not remaining:
        # Nothing is open: the issue was canceled only when EVERY card was
        # archived without completing. One completed card is real work
        # whatever dead duplicate was archived beside it, so that is Done
        # (ENG-2230, ENG-2234).
        if all(card.status == "archived" and not is_completed(card) for card in cards):
            return "Canceled"
        return done()

    if all(card.assignee == "reviewer" for card in remaining):
        return "Reviewing"
    return "Working"


def with_blocked(cards: list[Card], status: str | None) -> str | None:
    """The issue's status while any card is blocked, else the given status.

    Nothing is invented for an issue with no card.
    """
    return BLOCKED_STATE if any(card.status == "blocked" for card in cards) else status
