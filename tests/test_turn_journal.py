from __future__ import annotations

import sys
import time
import types

import pytest

from hermes_linear_agent import gateway_state as gs
from hermes_linear_agent.turn_journal import TurnJournal


@pytest.fixture
def journal(tmp_path):
    return TurnJournal(tmp_path / "state" / "turn_journal.json")


def test_queue_then_take_pending_in_order_and_clears(journal):
    journal.queue("i1", "ENG-1", "a")
    journal.queue("i2", "ENG-2", "b")
    journal.queue("i1", "ENG-1", "c")
    queued, interrupted = journal.take_pending()
    assert [e.text for e in queued] == ["a", "b", "c"]
    assert interrupted == []
    assert journal.take_pending() == ([], [])


def test_start_finish_clears(journal):
    journal.start("i1", "ENG-1", "t")
    journal.finish("i1")
    journal.finish("missing")
    assert journal.take_pending() == ([], [])


def test_start_twice_keeps_latest(journal):
    journal.start("i1", "ENG-1", "old")
    journal.start("i1", "ENG-1", "new")
    queued, interrupted = journal.take_pending()
    assert queued == []
    assert [e.text for e in interrupted] == ["new"]


def test_requeue(journal):
    assert journal.requeue("i1") is False
    journal.start("i1", "ENG-1", "hello")
    assert journal.requeue("i1") is True
    queued, interrupted = journal.take_pending()
    assert [(e.issue_id, e.issue_key, e.text) for e in queued] == [("i1", "ENG-1", "hello")]
    assert interrupted == []


def test_old_inflight_dropped(tmp_path):
    j = TurnJournal(tmp_path / "j.json", max_inflight_age_s=0.01)
    j.start("i1", "ENG-1", "t")
    time.sleep(0.05)
    assert j.take_pending() == ([], [])


def test_corrupt_file_reads_empty(tmp_path):
    path = tmp_path / "j.json"
    path.write_text("{not json")
    j = TurnJournal(path)
    assert j.take_pending() == ([], [])
    j.queue("i1", "ENG-1", "x")
    assert len(j.take_pending()[0]) == 1


def test_survives_new_instance(tmp_path):
    path = tmp_path / "j.json"
    first = TurnJournal(path)
    first.queue("i1", "ENG-1", "q")
    first.start("i2", "ENG-2", "f")
    queued, interrupted = TurnJournal(path).take_pending()
    assert [e.text for e in queued] == ["q"]
    assert [e.text for e in interrupted] == ["f"]


def _fake_status(monkeypatch, record, stale=False):
    mod = types.ModuleType("gateway.status")
    mod.read_runtime_status = lambda *a, **k: record
    mod.runtime_status_is_stale = lambda *a, **k: stale
    monkeypatch.setitem(sys.modules, "gateway", types.ModuleType("gateway"))
    monkeypatch.setitem(sys.modules, "gateway.status", mod)


def test_draining_fresh(monkeypatch):
    _fake_status(monkeypatch, {"gateway_state": "draining"})
    assert gs.gateway_draining() is True
    assert gs.gateway_ready() is False


def test_draining_false_when_stale_or_none(monkeypatch):
    _fake_status(monkeypatch, {"gateway_state": "draining"}, stale=True)
    assert gs.gateway_state() is None
    assert gs.gateway_draining() is False
    _fake_status(monkeypatch, None)
    assert gs.gateway_draining() is False


def test_draining_false_when_import_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "gateway.status", None)
    assert gs.gateway_draining() is False
    assert gs.gateway_ready() is False


@pytest.mark.parametrize("state", ["running", "degraded"])
def test_ready_states(monkeypatch, state):
    _fake_status(monkeypatch, {"gateway_state": state})
    assert gs.gateway_ready() is True


def test_not_ready_when_starting(monkeypatch):
    _fake_status(monkeypatch, {"gateway_state": "starting"})
    assert gs.gateway_ready() is False
