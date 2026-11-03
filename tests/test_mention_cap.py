from __future__ import annotations

from hermes_linear_agent.mention_cap import MentionCap


def test_limit_allows_then_refuses(tmp_path):
    cap = MentionCap(tmp_path / "m.json", 2)
    assert cap.allow("a") == (True, 1)
    assert cap.allow("a") == (True, 2)
    assert cap.allow("a") == (False, 3)
    assert cap.allow("a")[0] is False


def test_count_survives_new_instance(tmp_path):
    path = tmp_path / "sub" / "m.json"
    MentionCap(path, 2).allow("a")
    assert MentionCap(path, 2).allow("a") == (True, 2)
    assert MentionCap(path, 2).allow("a") == (False, 3)


def test_issues_count_separately(tmp_path):
    cap = MentionCap(tmp_path / "m.json", 1)
    assert cap.allow("a") == (True, 1)
    assert cap.allow("b") == (True, 1)
    assert cap.allow("a")[0] is False


def test_corrupt_file_reads_as_empty(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json", encoding="utf-8")
    assert MentionCap(path, 2).allow("a") == (True, 1)
