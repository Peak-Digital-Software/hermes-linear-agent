"""Tests for hermes_linear_agent.waiters: wake the delegated ancestor when a blocker closes."""

from __future__ import annotations

from typing import Any

from hermes_linear_agent import waiters
from hermes_linear_agent.waiters import Waiters, all_blockers_closed, blockers, nearest_delegated

APP = "app-user"


def blocker(bid: str, state: str = "completed") -> dict:
    return {"type": "blocks", "issue": {"id": bid, "identifier": bid.upper(), "state": {"type": state}}}


def issue(
    iid: str, blocked_by: list[dict], parent: dict | None = None, delegate: str | None = None, state: str = "started"
) -> dict:
    node: dict[str, Any] = {
        "id": iid,
        "identifier": iid.upper(),
        "state": {"type": state},
        "delegate": {"id": delegate} if delegate else None,
        "inverseRelations": {"nodes": blocked_by},
    }
    if parent is not None:
        node["parent"] = parent
    return node


def ancestor(iid: str, delegate: str | None = None, parent: dict | None = None) -> dict:
    node: dict[str, Any] = {"id": iid, "identifier": iid.upper(), "delegate": {"id": delegate} if delegate else None}
    if parent is not None:
        node["parent"] = parent
    return node


class FakeClient:
    def __init__(self, blocked: list[dict] | None = None, waiting: list[dict] | None = None, one: dict | None = None):
        self.blocked = blocked or []
        self.waiting = waiting or []
        self.one = one
        self.calls: list[str] = []

    async def execute(self, query: str, variables: dict | None = None) -> dict:
        name = query.split("(")[0].split()[1]
        self.calls.append(name)
        if name == "BlockedByMe":
            nodes = [{"type": "blocks", "relatedIssue": i} for i in self.blocked]
            return {"issue": {"id": (variables or {})["id"], "relations": {"nodes": nodes}}}
        if name == "Waiting":
            return {"issues": {"nodes": self.waiting, "pageInfo": {"hasNextPage": False, "endCursor": None}}}
        if name == "Nearest":
            return {"issue": self.one}
        raise AssertionError(name)


def make(client: FakeClient, open_cards: set[str] | None = None, app: str = APP):
    wakes: list[tuple[str, str, str]] = []

    async def has_open_card(key: str) -> bool:
        return key in (open_cards or set())

    return Waiters(client, app, lambda i, k, t: wakes.append((i, k, t)), has_open_card), wakes


def test_helpers():
    node = issue("x", [blocker("b1"), {"type": "related", "issue": {"id": "r"}}, blocker("b2", "started")])
    assert [b["id"] for b in blockers(node)] == ["b1", "b2"]
    assert not all_blockers_closed(node)
    assert all_blockers_closed(issue("x", [blocker("b1"), blocker("b2", "canceled")]))
    assert not all_blockers_closed(issue("x", []))
    assert nearest_delegated(issue("x", [], delegate=APP), APP) == ("x", "X")
    assert nearest_delegated(issue("x", [], delegate=APP), "") is None
    assert nearest_delegated(issue("x", [], parent=ancestor("p")), APP) is None
    assert waiters.CLOSED == {"completed", "canceled"}


async def test_undelegated_subissue_wakes_delegated_parent():
    sub = issue("sub", [blocker("b1")], parent=ancestor("par", APP))
    w, wakes = make(FakeClient(blocked=[sub]))
    assert await w.on_blocker_closed("b1") == ["PAR"]
    assert wakes == [("par", "PAR", "[sync] SUB unblocked.")]


async def test_grandparent_when_parent_undelegated():
    sub = issue("sub", [blocker("b1")], parent=ancestor("par", None, ancestor("gp", APP)))
    w, wakes = make(FakeClient(blocked=[sub]))
    assert await w.on_blocker_closed("b1") == ["GP"]
    assert wakes == [("gp", "GP", "[sync] SUB unblocked.")]


async def test_still_blocked_not_woken():
    sub = issue("sub", [blocker("b1"), blocker("b2", "started")], parent=ancestor("par", APP))
    w, wakes = make(FakeClient(blocked=[sub]))
    assert await w.on_blocker_closed("b1") == []
    assert wakes == []


async def test_closed_waiter_not_woken():
    sub = issue("sub", [blocker("b1")], parent=ancestor("par", APP), state="completed")
    w, wakes = make(FakeClient(blocked=[sub]))
    assert await w.on_blocker_closed("b1") == []
    assert wakes == []


async def test_repeat_event_wakes_nothing():
    sub = issue("sub", [blocker("b1")], parent=ancestor("par", APP))
    w, wakes = make(FakeClient(blocked=[sub]))
    assert await w.on_blocker_closed("b1") == ["PAR"]
    assert await w.on_blocker_closed("b1") == []
    assert len(wakes) == 1


async def test_empty_app_user_wakes_nothing():
    sub = issue("sub", [blocker("b1")], parent=ancestor("par", APP))
    client = FakeClient(blocked=[sub], waiting=[sub], one=sub)
    w, wakes = make(client, app="")
    assert await w.on_blocker_closed("b1") == []
    assert await w.reconcile(("ENG",)) == []
    assert await w.nearest_for("sub") is None
    assert wakes == []


async def test_reconcile_wakes_stale_waiter_without_open_card():
    sub = issue("sub", [blocker("b1")], parent=ancestor("par", APP))
    w, wakes = make(FakeClient(waiting=[sub]))
    assert await w.reconcile(("ENG",)) == ["PAR"]
    assert wakes == [("par", "PAR", "[sync] SUB unblocked.")]


async def test_reconcile_skips_open_card_and_still_blocked():
    held = issue("held", [blocker("b1")], parent=ancestor("par", APP))
    busy = issue("busy", [blocker("b1"), blocker("b2", "unstarted")], parent=ancestor("par", APP))
    w, wakes = make(FakeClient(waiting=[held, busy]), open_cards={"HELD"})
    assert await w.reconcile(("ENG",)) == []
    assert wakes == []


async def test_repeat_pass_wakes_nothing_and_event_then_pass_dedups():
    sub = issue("sub", [blocker("b1")], parent=ancestor("par", APP))
    w, wakes = make(FakeClient(blocked=[sub], waiting=[sub]))
    assert await w.on_blocker_closed("b1") == ["PAR"]
    assert await w.reconcile(("ENG",)) == []
    assert await w.reconcile(("ENG",)) == []
    assert len(wakes) == 1


async def test_failing_issue_logs_and_rest_continue(caplog):
    bad = issue("bad", [blocker("b1")], parent=ancestor("p1", APP))
    good = issue("good", [blocker("b1")], parent=ancestor("p2", APP))
    client = FakeClient(blocked=[bad, good])
    wakes: list[str] = []

    def wake(i: str, k: str, t: str) -> None:
        if i == "p1":
            raise RuntimeError("boom")
        wakes.append(k)

    async def none(_k: str) -> bool:
        return False

    w = Waiters(client, APP, wake, none)
    assert await w.on_blocker_closed("b1") == ["P2"]
    assert wakes == ["P2"]
    assert "boom" in caplog.text


async def test_nearest_for():
    sub = issue("sub", [], parent=ancestor("par", None, ancestor("gp", APP)))
    w, _ = make(FakeClient(one=sub))
    assert await w.nearest_for("sub") == ("gp", "GP")
    w2, _ = make(FakeClient(one=None))
    assert await w2.nearest_for("sub") is None


def test_queries_name_required_fields():
    for q in (waiters.BLOCKED_BY_ME, waiters.WAITING, waiters.NEAREST):
        assert "inverseRelations(first: 50)" in q
        assert q.count("parent {") == waiters.PARENT_DEPTH
    assert "pageInfo" in waiters.WAITING
