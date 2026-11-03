from __future__ import annotations

from pathlib import Path

from hermes_linear_agent import pr_reconcile
from hermes_linear_agent.github_prs import OpenPr
from hermes_linear_agent.kanban_rules import Card


def _pr(repo="alpha", number=7, title="ENG-100 fix thing", sha="abc123"):
    return OpenPr(
        repo=repo,
        full_name=f"Peak-Digital-Software/{repo}",
        number=number,
        url=f"https://github.com/Peak-Digital-Software/{repo}/pull/{number}",
        head_sha=sha,
        title=title,
        base="main",
    )


class Fakes:
    def __init__(self, prs_by_repo, cards=None, failing=()):
        self.prs_by_repo = prs_by_repo
        self.cards = cards or {}
        self.failing = set(failing)
        self.creates: list[tuple] = []
        self.clones: list[str] = []

    async def repos(self, gh, org):
        return [f"{org}/{name}" for name in self.prs_by_repo]

    async def prs(self, gh, full_name):
        name = full_name.split("/")[1]
        if name in self.failing:
            raise RuntimeError("boom")
        return self.prs_by_repo[name]

    async def clone(self, gh, full_name, root):
        self.clones.append(full_name)
        return root / full_name

    async def list_cards(self, board, key):
        return self.cards.get(key, [])

    async def create(self, board, title, **kw):
        self.creates.append((board, title, kw))
        return f"t_{len(self.creates)}"

    async def run(self):
        return await pr_reconcile.reconcile_prs(
            gh="gh",
            clone_root=Path("/clones"),
            list_cards=self.list_cards,
            create=self.create,
            repos=self.repos,
            prs=self.prs,
            clone=self.clone,
        )


async def test_pr_without_card_creates_one():
    f = Fakes({"alpha": [_pr()]})
    assert await f.run() == ["t_1"]
    ((board, title, kw),) = f.creates
    assert board == "engineering"
    assert title == "Oracle: review and merge alpha#7"
    assert kw["idempotency_key"] == "oracle/alpha#7@abc123"
    assert kw["tenant"] == "ENG-100"
    assert kw["assignee"] == "oracle"
    assert kw["workspace"] == "worktree:/clones/Peak-Digital-Software/alpha"
    assert kw["completion_contract"] == "local-only"


async def test_second_pass_with_open_card_creates_nothing():
    f = Fakes({"alpha": [_pr()]})
    await f.run()
    f.cards["ENG-100"] = [Card("t_1", "ready", "oracle", None)]
    assert await f.run() == []
    assert len(f.creates) == 1


async def test_open_ship_card_in_review_skips():
    f = Fakes({"alpha": [_pr()]}, cards={"ENG-100": [Card("t_9", "review", "coder", None)]})
    assert await f.run() == []
    assert f.creates == [] and f.clones == []


async def test_done_card_does_not_block():
    f = Fakes({"alpha": [_pr()]}, cards={"ENG-100": [Card("t_9", "done", "coder", 1)]})
    assert await f.run() == ["t_1"]


async def test_title_without_key_uses_pr_tenant():
    f = Fakes({"alpha": [_pr(title="no key here")]})
    await f.run()
    assert f.creates[0][2]["tenant"] == "pr:alpha#7"


def test_tenant_for_picks_first_key():
    assert pr_reconcile.tenant_for(_pr(title="OPS-12 then ENG-3")) == "OPS-12"
    assert pr_reconcile.tenant_for(_pr(title="x ENG-5 y")) == "ENG-5"


async def test_failing_repo_does_not_stop_others():
    f = Fakes({"alpha": [_pr()], "beta": [_pr(repo="beta", number=2, title="ENG-2 x")]}, failing=["alpha"])
    assert await f.run() == ["t_1"]
    assert f.creates[0][2]["tenant"] == "ENG-2"


async def test_failing_create_does_not_stop_other_prs():
    f = Fakes({"alpha": [_pr(), _pr(number=8, title="ENG-101 y")]})
    real = f.create
    calls = []

    async def flaky(board, title, **kw):
        calls.append(title)
        if len(calls) == 1:
            raise RuntimeError("nope")
        return await real(board, title, **kw)

    f.create = flaky
    assert await f.run() == ["t_1"]


def test_oracle_body_contents():
    body = pr_reconcile.oracle_body(_pr())
    for needle in (
        "https://github.com/Peak-Digital-Software/alpha/pull/7",
        "abc123",
        "Base: main",
        "production == false",
        "kanban_block",
        "peak.toml",
        "do not deploy",
        "gh pr merge 7 -R Peak-Digital-Software/alpha --auto --squash",
        "enablePullRequestAutoMerge",
        "--match-head-commit",
        "Never `--admin`",
        "merged sha",
    ):
        assert needle in body
