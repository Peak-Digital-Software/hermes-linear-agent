"""Give every open internal PR an Oracle card (one pass; the caller loops)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from . import github_prs, kanban_cli
from .kanban_rules import is_open

logger = logging.getLogger(__name__)

ORACLE_KEY = "oracle/{repo}#{number}@{sha}"

_ISSUE_RE = re.compile(r"\b(?:ENG|OPS)-\d+\b")


def tenant_for(pr: github_prs.OpenPr) -> str:
    """First ENG-<n> or OPS-<n> in the PR title, else ``pr:<repo>#<number>``."""
    match = _ISSUE_RE.search(pr.title or "")
    if match:
        return match.group(0)
    return f"pr:{pr.repo}#{pr.number}"


def oracle_body(pr: github_prs.OpenPr, org: str = "Peak-Digital-Software") -> str:
    owner = pr.full_name.split("/", 1)[0] if "/" in pr.full_name else org
    return f"""Review and merge {pr.full_name}#{pr.number}: {pr.title}

- PR: {pr.url}
- Head sha: {pr.head_sha}
- Base: {pr.base}

Repo is production == false (re-check `gh api orgs/{owner}/properties/values` before merging; if it reads true, stop and kanban_block).

Test command: the repo's `peak.toml` `suite`, else the test/lint lines in `.github/workflows/*`.

Approve and merge, do not deploy.

Merge with:
`gh pr merge {pr.number} -R {pr.full_name} --auto --squash`
If that errors with `enablePullRequestAutoMerge`, wait for `ci` green and run:
`gh pr merge {pr.number} -R {pr.full_name} --squash --delete-branch --match-head-commit <reviewed_sha>`

Never `--admin`.

Report the merged sha in summary.
"""


async def reconcile_prs(
    *,
    gh: str,
    org: str = "Peak-Digital-Software",
    board: str = "engineering",
    clone_root: Path = Path("/home/db/.local/state/linear-agent/repos"),
    list_cards=kanban_cli.cards,
    create=kanban_cli.create,
    repos=github_prs.internal_repos,
    prs=github_prs.open_prs,
    clone=github_prs.ensure_clone,
) -> list[str]:
    """One pass over every internal repo's open PRs; returns created card ids."""
    created: list[str] = []
    try:
        full_names = await repos(gh, org)
    except Exception:
        logger.exception("pr reconcile: listing internal repos failed")
        return created
    for full_name in full_names:
        try:
            open_prs = await prs(gh, full_name)
        except Exception:
            logger.exception("pr reconcile: listing PRs of %s failed", full_name)
            continue
        for pr in open_prs:
            try:
                tenant = tenant_for(pr)
                existing = await list_cards(board, tenant)
                if any(is_open(card) for card in existing):
                    continue
                path = await clone(gh, pr.full_name, clone_root)
                card_id = await create(
                    board,
                    f"Oracle: review and merge {pr.repo}#{pr.number}",
                    body=oracle_body(pr, org),
                    assignee="oracle",
                    tenant=tenant,
                    idempotency_key=ORACLE_KEY.format(repo=pr.repo, number=pr.number, sha=pr.head_sha),
                    workspace=f"worktree:{path}",
                    completion_contract="local-only",
                )
                created.append(card_id)
            except Exception:
                logger.exception("pr reconcile: %s#%s failed", pr.full_name, pr.number)
    return created
