"""Pure status rules for the kanban <-> Linear bridge.

Ported case-for-case from peak-agent-tools' TypeScript suites:
``issuesync.test.ts`` (transition table, closed types, open cards) and
``watch.test.ts`` (issueStatus, prGate), plus the ENG-2286 8-row
``issueStatus`` precedence and the parent-session ``rollup``. The tables are
the specification; keep them in sync.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from hermes_linear_agent.kanban_rules import (
    ACTED,
    CLOSED_TYPES,
    STATE_IDS,
    Card,
    is_completed,
    is_open,
    issue_status,
    pr_gate,
    rollup,
    transition,
)


def card(
    card_id: str,
    status: str,
    assignee: str | None = None,
    completed_at: int | None = None,
) -> Card:
    return Card(id=card_id, status=status, assignee=assignee, completed_at=completed_at)


def pr_link(number: int, status: str | None = None) -> dict:
    metadata: dict = {"number": number}
    if status is not None:
        metadata["status"] = status
    return {"url": f"https://github.com/x/y/pull/{number}", "sourceType": "github", "metadata": metadata}


REPO_LINK = {
    "url": "https://github.com/Peak-Digital-Software/peak-agent-tools",
    "sourceType": "api",
    "metadata": {},
}


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_the_closed_set_is_exactly_completed_canceled_and_duplicate():
    assert sorted(CLOSED_TYPES) == ["canceled", "completed", "duplicate"]


def test_acted_is_the_watch_event_set():
    assert ACTED == frozenset(
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
    # noise events never reach Linear
    for noise in ("heartbeat", "worker_bound", "dependency_wait"):
        assert noise not in ACTED


# ---------------------------------------------------------------------------
# transition (issuesync.ts:21-63, issuesync.test.ts:55-103)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "was_type", "type_", "expected"),
    [
        # creation and non-moves are never a card action
        ("create", None, "backlog", "none"),
        ("create", None, "unstarted", "none"),
        ("create", None, "triage", "none"),
        ("create", None, "started", "none"),
        ("create", None, None, "none"),
        ("update", None, "backlog", "none"),
        ("update", "started", "started", "none"),
        ("update", None, None, "none"),
        ("remove", "started", "completed", "none"),
        # a Triage move creates and closes nothing but Done
        ("update", "triage", "backlog", "none"),
        ("update", "triage", "unstarted", "none"),
        ("update", "triage", "started", "none"),
        ("update", "triage", "completed", "close"),
        ("update", "triage", "canceled", "none"),
        ("update", "triage", "duplicate", "none"),
        # a non-closed issue moved to Canceled or Duplicate is canceled
        ("update", "backlog", "canceled", "cancel"),
        ("update", "backlog", "duplicate", "cancel"),
        ("update", "unstarted", "canceled", "cancel"),
        ("update", "unstarted", "duplicate", "cancel"),
        ("update", "started", "canceled", "cancel"),
        ("update", "started", "duplicate", "cancel"),
        ("update", "completed", "canceled", "none"),
        # a non-completed issue reaching Done is closed
        ("update", "backlog", "completed", "close"),
        ("update", "started", "completed", "close"),
        ("update", "completed", "completed", "none"),
        # a closed issue moved back to an open type is no card action
        ("update", "completed", "backlog", "none"),
        ("update", "completed", "unstarted", "none"),
        ("update", "completed", "started", "none"),
        ("update", "canceled", "backlog", "none"),
        ("update", "canceled", "unstarted", "none"),
        ("update", "canceled", "started", "none"),
        ("update", "duplicate", "backlog", "none"),
        ("update", "duplicate", "unstarted", "none"),
        ("update", "duplicate", "started", "none"),
        ("update", "completed", "triage", "none"),
    ],
)
def test_transition_maps_every_state_move_to_cancel_close_or_nothing(action, was_type, type_, expected):
    assert transition(action, was_type, type_) == expected


# ---------------------------------------------------------------------------
# is_open / is_completed (issuesync.ts, watch.ts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["ready", "running", "blocked", "review", "todo"])
def test_an_open_card_can_still_change(status):
    assert is_open(card("t1", status)) is True


@pytest.mark.parametrize("status", ["done", "archived"])
def test_only_done_and_archived_cards_are_closed(status):
    assert is_open(card("t1", status)) is False


@pytest.mark.parametrize(
    ("status", "completed_at", "expected"),
    [
        ("done", None, True),
        ("done", 5, True),
        ("running", 3, True),
        ("archived", 2, True),
        ("running", None, False),
        ("archived", None, False),
        ("ready", None, False),
    ],
)
def test_a_card_is_completed_by_status_or_completion_time(status, completed_at, expected):
    assert is_completed(card("t1", status, completed_at=completed_at)) is expected


# ---------------------------------------------------------------------------
# pr_gate (watch.ts, watch.test.ts ENG-2239)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        pytest.param([], "none", id="no-attachments-at-all"),
        pytest.param([REPO_LINK], "none", id="only-the-repo-link"),
        pytest.param(
            [{"url": "https://github.com/x/y", "sourceType": "github"}],
            "none",
            id="github-link-without-metadata-number",
        ),
        pytest.param(
            [{"url": "https://github.com/x/y", "sourceType": "github", "metadata": None}],
            "none",
            id="github-link-with-null-metadata",
        ),
        pytest.param([REPO_LINK, pr_link(156, "open")], "open", id="an-open-pr"),
        pytest.param([pr_link(156, "draft")], "open", id="a-draft-pr"),
        pytest.param([pr_link(156, "review")], "open", id="a-pr-in-review"),
        pytest.param([pr_link(156)], "open", id="a-pr-with-no-status-yet"),
        pytest.param([pr_link(156, "merged")], "merged", id="a-merged-pr"),
        pytest.param([pr_link(156, "closed")], "merged", id="a-closed-pr-without-merge"),
        pytest.param([pr_link(1, "merged"), pr_link(2, "closed")], "merged", id="merged-and-closed"),
        pytest.param([pr_link(1, "merged"), pr_link(2, "open")], "open", id="one-merged-one-open"),
    ],
)
def test_pr_gate_reads_open_merged_and_none_from_attachments(links, expected):
    assert pr_gate(links) == expected


# ---------------------------------------------------------------------------
# STATE_IDS (ENG-2286)
# ---------------------------------------------------------------------------


def test_state_ids_are_the_eng_team_workflow_state_ids():
    assert STATE_IDS == {
        "Blocked": "8de0dfd6-614f-4d0d-9ee1-aee47a12bb58",
        "Working": "f41a9dcb-10c9-41d9-a7cf-e83f0e49367a",
        "Reviewing": "916a7971-5640-4989-a7ae-d8175499851f",
        "Ready to Merge": "40c4ac3a-e45a-4817-a19b-7b94eb1bb0ed",
        "Todo": "bb1c80eb-1a31-4616-9033-602a12b82ba0",
        "Done": "975dfb59-166d-4df4-b70c-5382442448d8",
        "Canceled": "7fc0e9b4-2d39-45dc-87aa-79a8d9c5ce5c",
    }


# ---------------------------------------------------------------------------
# issue_status (ENG-2286: 8-row precedence, first match wins)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        # row 1: any blocked card wins over everything
        pytest.param([card("t_a", "blocked")], "Blocked", id="row1-one-blocked"),
        pytest.param([card("t_a", "blocked"), card("t_b", "running")], "Blocked", id="row1-blocked-beats-running"),
        pytest.param([card("t_a", "blocked"), card("t_b", "todo")], "Blocked", id="row1-blocked-beats-todo"),
        pytest.param([card("t_a", "blocked"), card("t_b", "review")], "Blocked", id="row1-blocked-beats-review"),
        # row 2: any running card wins over review and everything below
        pytest.param([card("t_a", "running")], "Working", id="row2-one-running"),
        pytest.param([card("t_a", "running"), card("t_b", "review")], "Working", id="row2-running-beats-review"),
        pytest.param([card("t_a", "running"), card("t_b", "todo")], "Working", id="row2-running-beats-todo"),
        pytest.param([card("t_a", "running", completed_at=3)], "Working", id="row2-running-beats-completed-at"),
        # row 3: any review card
        pytest.param([card("t_a", "review")], "Reviewing", id="row3-one-review"),
        pytest.param([card("t_a", "review"), card("t_b", "todo")], "Reviewing", id="row3-review-beats-todo"),
        pytest.param([card("t_a", "review"), card("t_b", "ready")], "Reviewing", id="row3-review-beats-ready"),
        # row 4: every open card is todo/ready/triage
        pytest.param([card("t_a", "todo")], "Todo", id="row4-one-todo"),
        pytest.param([card("t_a", "ready")], "Todo", id="row4-one-ready"),
        pytest.param([card("t_a", "triage")], "Todo", id="row4-one-triage"),
        pytest.param(
            [card("t_a", "done", completed_at=1), card("t_b", "todo")], "Todo", id="row4-done-plus-todo-is-todo"
        ),
        pytest.param(
            [card("t_a", "done", completed_at=1), card("t_b", "todo"), card("t_c", "ready")],
            "Todo",
            id="row4-done-plus-todo-and-ready-is-todo",
        ),
        # row 5: every card completed, PR still open
        pytest.param(
            [card("t_a", "done", completed_at=1), card("t_b", "done", completed_at=2)],
            "Ready to Merge",
            id="row5-done-set-with-open-pr",
        ),
        # row 6: every card completed, PR merged or none
        pytest.param(
            [card("t_a", "done", completed_at=1), card("t_b", "done", completed_at=2)],
            "Done",
            id="row6-done-set-with-no-pr",
        ),
        pytest.param(
            [card("t_a", "done", completed_at=1), card("t_b", "archived")],
            "Done",
            id="row6-done-plus-archived-never-completed-is-done",
        ),
        pytest.param(
            [card("t_a", "done", completed_at=1), card("t_b", "archived", completed_at=2)],
            "Done",
            id="row6-done-plus-archived-completed-is-done",
        ),
        # row 7: every card archived and none completed
        pytest.param([card("t_a", "archived")], "Canceled", id="row7-one-archived-never-completed"),
        pytest.param(
            [card("t_a", "archived"), card("t_b", "archived")],
            "Canceled",
            id="row7-two-archived-never-completed",
        ),
        # row 8: no cards
        pytest.param([], None, id="row8-no-cards"),
        # leftover: an open card that is none of the above behaves as work
        pytest.param([card("t_a", "weird")], "Working", id="unknown-open-status-is-working"),
        pytest.param([card("t_a", "archived"), card("t_b", "running")], "Working", id="archived-beside-running"),
    ],
)
def test_issue_status_follows_the_eight_row_precedence(cards, expected):
    attachments = [pr_link(156, "open")] if expected == "Ready to Merge" else None
    assert issue_status(cards, attachments) == expected


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        pytest.param([], "Done", id="completed-set-with-no-pr"),
        pytest.param([REPO_LINK], "Done", id="completed-set-with-only-a-repo-link"),
        pytest.param([REPO_LINK, pr_link(156, "open")], "Ready to Merge", id="completed-set-with-an-open-pr"),
        pytest.param([pr_link(156, "merged")], "Done", id="completed-set-with-a-merged-pr"),
        pytest.param([pr_link(1, "merged"), pr_link(2, "open")], "Ready to Merge", id="one-open-pr-holds-ready"),
        pytest.param(
            [pr_link(1, "merged")],
            "Done",
            id="done-plus-archived-with-merged-pr-is-done",
        ),
    ],
)
def test_a_completed_card_set_is_ready_to_merge_only_when_no_pr_is_merged(links, expected):
    done_set = [card("t_a", "done", completed_at=1), card("t_b", "done", completed_at=2)]
    if links == [pr_link(1, "merged")]:
        done_set = [card("t_a", "done", completed_at=1), card("t_b", "archived")]
    assert issue_status(done_set, links) == expected


def test_attachments_default_to_none_and_never_gate_a_non_done_status():
    assert issue_status([card("t_a", "running")]) == "Working"
    assert issue_status([card("t_a", "running")], None) == "Working"
    # an open PR beside open work still means Working, not Ready to Merge
    assert issue_status([card("t_a", "running")], [pr_link(1, "open")]) == "Working"


# ---------------------------------------------------------------------------
# rollup (ENG-2286: parent card status from its child cards)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("child_states", "expected"),
    [
        pytest.param(["Blocked"], "Blocked", id="one-blocked"),
        pytest.param(["Working", "Blocked"], "Blocked", id="blocked-beats-working"),
        pytest.param(["Done", "Blocked"], "Blocked", id="blocked-beats-done"),
        pytest.param(["Working"], "Working", id="one-working"),
        pytest.param(["Reviewing"], "Working", id="reviewing-rolls-up-to-working"),
        pytest.param(["Ready to Merge"], "Working", id="ready-to-merge-rolls-up-to-working"),
        pytest.param(["Done", "Reviewing"], "Working", id="one-started-among-done"),
        pytest.param(["Done"], None, id="all-done"),
        pytest.param(["Done", "Canceled"], None, id="done-and-canceled"),
        pytest.param(["Canceled"], None, id="all-canceled"),
        pytest.param([], None, id="empty"),
        pytest.param(["Todo"], None, id="todo-only"),
        pytest.param(["Backlog"], None, id="backlog-only"),
        pytest.param(["Todo", "Backlog"], None, id="todo-and-backlog-only"),
    ],
)
def test_rollup_reads_blocked_then_working_and_otherwise_leaves_the_parent_alone(child_states, expected):
    assert rollup(child_states) == expected


def test_cards_are_frozen():
    original = card("t_a", "done")
    with pytest.raises(FrozenInstanceError):
        original.status = "running"  # type: ignore[misc]
