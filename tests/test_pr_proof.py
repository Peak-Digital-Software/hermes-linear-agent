from __future__ import annotations

import json

from hermes_linear_agent import github_prs, pr_proof
from hermes_linear_agent.github_prs import OpenPr
from hermes_linear_agent.kanban_rules import Card

ORG = "Peak-Digital-Software"


def _pr(repo="alpha", number=7, title="ENG-100 fix thing", sha="abcdef1234567890"):
    return OpenPr(
        repo=repo,
        full_name=f"{ORG}/{repo}",
        number=number,
        url=f"https://github.com/{ORG}/{repo}/pull/{number}",
        head_sha=sha,
        title=title,
        base="main",
        branch=f"feat-{number}",
    )


class Fakes:
    def __init__(self, prs_by_repo, files=None, production=None, failing=()):
        self.prs_by_repo = prs_by_repo
        self.files = files or {}
        self.production = production if production is not None else {name: True for name in prs_by_repo}
        self.failing = set(failing)
        self.cards: dict[str, dict[str, Card]] = {}  # tenant -> id -> Card
        self.by_key: dict[str, str] = {}
        self.creates: list[tuple] = []

    async def repos(self, gh, org):
        return {f"{org}/{name}": flag for name, flag in self.production.items()}

    async def prs(self, gh, full_name):
        name = full_name.split("/")[1]
        if name in self.failing:
            raise RuntimeError("boom")
        return self.prs_by_repo[name]

    async def changed(self, gh, full_name, number):
        return self.files.get((full_name.split("/")[1], number), ["x.py"])

    async def list_cards(self, board, key):
        return list(self.cards.get(key, {}).values())

    async def create(self, board, title, **kw):
        key = kw["idempotency_key"]
        if key in self.by_key:  # kanban returns the existing card for a repeated key
            return self.by_key[key]
        card_id = f"t_{len(self.by_key) + 1}"
        self.by_key[key] = card_id
        self.creates.append((board, title, kw))
        self.cards.setdefault(kw["tenant"], {})[card_id] = Card(card_id, "ready", kw["assignee"], None)
        return card_id

    def finish(self, card_id):
        for cards in self.cards.values():
            if card_id in cards:
                cards[card_id] = Card(card_id, "done", "visual-proof", 1)

    async def run(self):
        return await pr_proof.reconcile_proofs(
            gh="gh",
            list_cards=self.list_cards,
            create=self.create,
            repos=self.repos,
            prs=self.prs,
            files=self.changed,
        )


async def test_ui_pr_on_production_repo_creates_one_card():
    f = Fakes({"alpha": [_pr()]}, files={("alpha", 7): ["src/Button.tsx"]})
    assert await f.run() == ["t_1"]
    ((board, title, kw),) = f.creates
    assert board == "engineering"
    assert title == "Visual proof: alpha#7 @ abcdef1"
    assert kw["idempotency_key"] == "proof/alpha#7@abcdef1234567890"
    assert kw["tenant"] == "proof:alpha"
    assert kw["assignee"] == "visual-proof"
    assert kw["skills"] == ["visual-proof"]
    assert kw["workspace"] == "scratch"
    assert kw["completion_contract"] == "local-only"
    body = kw["body"]
    assert "ENG-100" in body
    assert "peak-preview up alpha --ref feat-7 --json" in body
    assert "Head sha: abcdef1234567890" in body
    assert "Follow skill visual-proof." in body
    assert "PNG" not in body and "steps comment" not in body


async def test_second_pass_creates_nothing():
    f = Fakes({"alpha": [_pr()]}, files={("alpha", 7): ["a.tsx"]})
    await f.run()
    assert await f.run() == []
    f.finish("t_1")
    assert await f.run() == []
    assert len(f.creates) == 1


async def test_ui_path_detection():
    for path in ("a.tsx", "a.jsx", "a.css", "a.html", "src/app/page.ts", "components/x.ts"):
        assert pr_proof.is_ui(path), path
    for path in ("a.py", "apps/x.py", "my-app/readme.md"):
        assert not pr_proof.is_ui(path), path
    f = Fakes(
        {"alpha": [_pr(number=1), _pr(number=2), _pr(number=3)]},
        files={("alpha", 1): ["a.py"], ("alpha", 2): ["apps/x.py", "my-app/readme.md"], ("alpha", 3): ["c.py"]},
    )
    assert await f.run() == []
    for number, path in ((4, "src/app/page.ts"), (5, "components/x.ts")):
        g = Fakes({"alpha": [_pr(number=number)]}, files={("alpha", number): [path]})
        assert len(await g.run()) == 1


async def test_non_production_repo_is_skipped():
    f = Fakes({"alpha": [_pr()]}, files={("alpha", 7): ["a.tsx"]}, production={"alpha": False})
    assert await f.run() == []
    assert f.creates == []


async def test_one_proof_per_repo_per_pass_and_next_after_done():
    f = Fakes(
        {"alpha": [_pr(number=9, sha="9" * 40), _pr(number=7, sha="7" * 40)]},
        files={("alpha", 7): ["a.tsx"], ("alpha", 9): ["b.tsx"]},
    )
    assert await f.run() == ["t_1"]
    assert f.creates[0][1].startswith("Visual proof: alpha#7")
    assert await f.run() == []  # first card still open: whole repo skipped
    f.finish("t_1")
    assert await f.run() == ["t_2"]
    assert f.creates[1][1].startswith("Visual proof: alpha#9")


async def test_new_head_sha_gets_a_new_card():
    f = Fakes({"alpha": [_pr(sha="a" * 40)]}, files={("alpha", 7): ["a.tsx"]})
    await f.run()
    f.finish("t_1")
    assert await f.run() == []
    f.prs_by_repo["alpha"] = [_pr(sha="b" * 40)]
    assert await f.run() == ["t_2"]
    assert f.creates[1][2]["idempotency_key"] == "proof/alpha#7@" + "b" * 40


async def test_failing_repo_does_not_stop_another():
    f = Fakes(
        {"alpha": [_pr()], "beta": [_pr(repo="beta")]},
        files={("alpha", 7): ["a.tsx"], ("beta", 7): ["a.tsx"]},
        failing={"alpha"},
    )
    assert await f.run() == ["t_1"]
    assert f.creates[0][2]["tenant"] == "proof:beta"


async def test_pr_without_issue_key_says_post_on_pr_only():
    f = Fakes({"alpha": [_pr(title="no key here")]}, files={("alpha", 7): ["a.tsx"]})
    await f.run()
    assert "none: PR has no issue key, block" in f.creates[0][2]["body"]


# --- github_prs additions ---------------------------------------------------


class _Proc:
    def __init__(self, out):
        self.out = out.encode()
        self.returncode = 0

    async def communicate(self):
        return self.out, b""


def _install(monkeypatch, out):
    calls = []

    async def fake(*argv, **kwargs):
        calls.append(argv)
        return _Proc(out)

    monkeypatch.setattr(github_prs.asyncio, "create_subprocess_exec", fake)
    return calls


async def test_repos_by_production(monkeypatch):
    rows = [
        {"repository_full_name": "o/a", "properties": [{"property_name": "production", "value": "true"}]},
        {"repository_full_name": "o/b", "properties": [{"property_name": "production", "value": "false"}]},
        {"repository_full_name": "o/c", "properties": []},
    ]
    calls = _install(monkeypatch, json.dumps(rows))
    assert await github_prs.repos_by_production("gh", "o") == {"o/a": True, "o/b": False, "o/c": False}
    assert calls == [("gh", "api", "orgs/o/properties/values", "--paginate")]


async def test_changed_files_argv(monkeypatch):
    calls = _install(monkeypatch, "a.tsx\nb/c.py\n")
    assert await github_prs.changed_files("gh", "o/r", 5) == ["a.tsx", "b/c.py"]
    assert calls == [("gh", "api", "repos/o/r/pulls/5/files", "--paginate", "--jq", ".[].filename")]
