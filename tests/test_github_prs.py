"""Tests for github_prs: gh is faked at the subprocess boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_linear_agent import github_prs

FIXTURES = Path(__file__).parent / "fixtures" / "github"


class FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout.encode()
        self.stderr = stderr.encode()
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr


class FakeExec:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.proc = FakeProc(stdout, stderr, returncode)
        self.calls: list[tuple] = []

    async def __call__(self, *argv, **kwargs):
        self.calls.append(argv)
        return self.proc


def install(monkeypatch, fake: FakeExec) -> FakeExec:
    monkeypatch.setattr(github_prs.asyncio, "create_subprocess_exec", fake)
    return fake


async def test_internal_repos_only_false(monkeypatch):
    rows = json.loads((FIXTURES / "properties.json").read_text())
    rows.append({"repository_full_name": "Peak-Digital-Software/no-prop", "properties": []})
    fake = install(monkeypatch, FakeExec(json.dumps(rows)))
    names = await github_prs.internal_repos("/bin/gh", "Peak-Digital-Software")
    assert fake.calls[0] == ("/bin/gh", "api", "orgs/Peak-Digital-Software/properties/values", "--paginate")
    assert "Peak-Digital-Software/hermes-linear-agent" in names
    assert "Peak-Digital-Software/cellarius" not in names  # "true"
    assert "Peak-Digital-Software/no-prop" not in names  # missing property
    expected = [
        r["repository_full_name"]
        for r in rows
        if any(p["property_name"] == "production" and p["value"] == "false" for p in r["properties"])
    ]
    assert names == expected
    assert len(names) == 11


async def test_internal_repos_handles_concatenated_pages(monkeypatch):
    page1 = json.dumps(
        [{"repository_full_name": "o/a", "properties": [{"property_name": "production", "value": "false"}]}]
    )
    page2 = json.dumps(
        [{"repository_full_name": "o/b", "properties": [{"property_name": "production", "value": "false"}]}]
    )
    install(monkeypatch, FakeExec(page1 + page2))
    assert await github_prs.internal_repos("gh", "o") == ["o/a", "o/b"]


async def test_open_prs_parses_fixture_and_skips_drafts(monkeypatch):
    rows = json.loads((FIXTURES / "pr-list.json").read_text())
    rows.append({**rows[0], "number": 99, "isDraft": True})
    fake = install(monkeypatch, FakeExec(json.dumps(rows)))
    prs = await github_prs.open_prs("gh", "Peak-Digital-Software/hermes-linear-agent")
    assert fake.calls[0] == (
        "gh",
        "pr",
        "list",
        "-R",
        "Peak-Digital-Software/hermes-linear-agent",
        "--state",
        "open",
        "--json",
        "number,url,headRefOid,title,isDraft,baseRefName",
    )
    assert [p.number for p in prs] == [22, 21]
    assert prs[0] == github_prs.OpenPr(
        repo="hermes-linear-agent",
        full_name="Peak-Digital-Software/hermes-linear-agent",
        number=22,
        url="https://github.com/Peak-Digital-Software/hermes-linear-agent/pull/22",
        head_sha="0431286857c5e040a0b87bbf1424fa22db99a267",
        title="ENG-2340: Linear sessions survive a gateway restart and take peer comments",
        base="main",
    )


async def test_ensure_clone_skips_when_git_exists(monkeypatch, tmp_path):
    (tmp_path / "o" / "r" / ".git").mkdir(parents=True)
    fake = install(monkeypatch, FakeExec())
    path = await github_prs.ensure_clone("gh", "o/r", tmp_path)
    assert path == tmp_path / "o" / "r"
    assert fake.calls == []


async def test_ensure_clone_builds_exact_argv(monkeypatch, tmp_path):
    fake = install(monkeypatch, FakeExec())
    path = await github_prs.ensure_clone("/bin/gh", "o/r", tmp_path)
    assert path == tmp_path / "o" / "r"
    assert fake.calls == [("/bin/gh", "repo", "clone", "o/r", str(tmp_path / "o" / "r"))]


async def test_nonzero_exit_raises(monkeypatch, tmp_path):
    install(monkeypatch, FakeExec(stderr="boom", returncode=1))
    with pytest.raises(RuntimeError, match="boom"):
        await github_prs.internal_repos("gh", "o")
    with pytest.raises(RuntimeError, match="boom"):
        await github_prs.open_prs("gh", "o/r")
    with pytest.raises(RuntimeError, match="boom"):
        await github_prs.ensure_clone("gh", "o/r", tmp_path)
