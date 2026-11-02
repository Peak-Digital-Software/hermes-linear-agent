"""Pure status rules for the kanban <-> Linear bridge.

Ported case-for-case from peak-agent-tools' TypeScript suites:
``issuesync.test.ts`` (transition table, closed types, open cards),
``blocked.test.ts`` (withBlocked), and ``watch.test.ts`` (issueStatus,
prGate). The tables are the specification; keep them in sync with the TS.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from hermes_linear_agent.kanban_rules import (
    ACTED,
    CLOSED_TYPES,
    Card,
    is_completed,
    is_open,
    issue_status,
    pr_gate,
    transition,
    with_blocked,
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
# with_blocked (blocked.ts:10-12, blocked.test.ts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cards", "status", "expected"),
    [
        pytest.param([card("t_a", "blocked"), card("t_b", "running")], "Working", "Blocked", id="one-blocked-wins"),
        pytest.param([card("t_a", "blocked")], "Blocked", "Blocked", id="already-blocked"),
        pytest.param(
            [card("t_a", "blocked"), card("t_b", "todo")], "Reviewing", "Blocked", id="blocked-beats-reviewing"
        ),
        pytest.param([card("t_a", "running")], "Working", "Working", id="no-blocked-card"),
        pytest.param([], None, None, id="no-cards-keeps-null"),
        pytest.param([], "Done", "Done", id="no-cards-keeps-given-status"),
        pytest.param([card("t_a", "done")], "Working", "Working", id="done-is-not-a-block"),
        pytest.param([card("t_a", "archived")], "Canceled", "Canceled", id="archived-is-not-a-block"),
    ],
)
def test_with_blocked_reads_blocked_from_any_blocked_card_and_only_then(cards, status, expected):
    assert with_blocked(cards, status) == expected


# ---------------------------------------------------------------------------
# issue_status (watch.ts:118-146, watch.test.ts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        pytest.param([], None, id="no-cards"),
        pytest.param([card("t_a", "todo", "coder")], "Working", id="one-open-card"),
        pytest.param([card("t_a", "done", "coder", 1), card("t_b", "done", "coder", 2)], "Done", id="everything-done"),
        pytest.param(
            [card("t_a", "done", "coder"), card("t_b", "running", "coder", 3)],
            "Done",
            id="done-by-completed_at-even-when-the-status-lags",
        ),
        pytest.param(
            [card("t_a", "done", "coder", 1), card("t_b", "archived", "coder", 2)],
            "Done",
            id="mixed-done-plus-archived-completed-is-done",
        ),
        pytest.param(
            [card("t_a", "done", "coder", 1), card("t_b", "archived", "coder")],
            "Done",
            id="one-completed-plus-archived-never-completed-is-done",
        ),
        pytest.param([card("t_a", "archived", "coder")], "Canceled", id="single-archived-not-completed"),
        pytest.param(
            [
                card("t_a", "done", "coder", 1),
                card("t_b", "todo", "reviewer"),
                card("t_c", "ready", "reviewer"),
            ],
            "Reviewing",
            id="every-open-card-is-the-reviewers",
        ),
        pytest.param(
            [card("t_a", "todo", "reviewer"), card("t_b", "ready", "coder")],
            "Working",
            id="one-open-coder-among-reviewers-is-working",
        ),
        pytest.param([card("t_a", "ready")], "Working", id="open-card-with-no-assignee-is-working"),
        pytest.param(
            [card("t_a", "archived"), card("t_b", "running", "coder")],
            "Working",
            id="archived-card-does-not-stop-open-work",
        ),
    ],
)
def test_issue_status_follows_the_status_rule_for_the_whole_card_set(cards, expected):
    assert issue_status(cards) == expected


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        pytest.param([], None, id="no-cards-at-all"),
        pytest.param([card("t_a", "archived", "orchestrator")], "Canceled", id="one-archived-never-completed"),
        pytest.param(
            [card("t_a", "archived", "orchestrator"), card("t_b", "archived", "orchestrator")],
            "Canceled",
            id="two-archived-never-completed",
        ),
        pytest.param([card("t_a", "done", "coder", 1)], "Done", id="one-completed"),
        pytest.param([card("t_a", "done", "coder", 1), card("t_b", "done", "coder", 2)], "Done", id="two-completed"),
        pytest.param(
            [card("t_a", "done", "coder", 1), card("t_b", "archived", "orchestrator")],
            "Done",
            id="completed-then-archived-never-completed",
        ),
        pytest.param(
            [card("t_a", "archived", "orchestrator"), card("t_b", "done", "coder", 1)],
            "Done",
            id="archived-never-completed-then-completed",
        ),
        pytest.param([card("t_a", "todo", "reviewer")], "Reviewing", id="open-reviewer-card-only"),
        pytest.param(
            [
                card("t_a", "done", "coder", 1),
                card("t_b", "archived", "orchestrator"),
                card("t_c", "ready", "reviewer"),
            ],
            "Reviewing",
            id="open-reviewer-beside-archived-and-completed",
        ),
        pytest.param(
            [card("t_a", "archived", "orchestrator"), card("t_b", "running", "coder")],
            "Working",
            id="open-coder-beside-archived-dead-is-working-never-canceled",
        ),
    ],
)
def test_issue_status_is_canceled_only_when_every_card_is_archived_and_uncompleted(cards, expected):
    assert issue_status(cards) == expected


@pytest.mark.parametrize(
    ("links", "expected"),
    [
        pytest.param([], "Done", id="completed-set-with-no-pr"),
        pytest.param([REPO_LINK], "Done", id="completed-set-with-only-a-repo-link"),
        pytest.param([REPO_LINK, pr_link(156, "open")], "Reviewing", id="completed-set-with-an-open-pr"),
        pytest.param([pr_link(156, "merged")], "Done", id="completed-set-with-a-merged-pr"),
        pytest.param([pr_link(1, "merged"), pr_link(2, "open")], "Reviewing", id="one-open-pr-holds-reviewing"),
    ],
)
def test_a_completed_card_set_reaches_done_only_when_no_pr_is_open(links, expected):
    done_set = [card("t_a", "done", "coder", 1), card("t_b", "done", "reviewer", 2)]
    assert issue_status(done_set, links) == expected


def test_attachments_default_to_none_and_never_gate_a_non_done_status():
    assert issue_status([card("t_a", "running", "coder")]) == "Working"
    assert issue_status([card("t_a", "running", "coder")], None) == "Working"
    # an open PR beside open work still means Working, not Reviewing
    assert issue_status([card("t_a", "running", "coder")], [pr_link(1, "open")]) == "Working"


def test_cards_are_frozen():
    original = card("t_a", "done")
    with pytest.raises(FrozenInstanceError):
        original.status = "running"  # type: ignore[misc]
