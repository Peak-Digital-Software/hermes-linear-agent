"""Give each UI PR head on a production repo one visual-proof card (one pass; the caller loops).

Kept apart from ``pr_reconcile`` so that module stays Oracle-only. Proofs are serialised
per repo (tenant ``proof:<repo>``) because previews share ports: a repo with an open proof
card gets nothing new this pass, and at most one new card per repo per pass.
"""

from __future__ import annotations

import logging
import re

from . import github_prs, kanban_cli
from .kanban_rules import is_open

logger = logging.getLogger(__name__)

UI_EXT = (".tsx", ".jsx", ".css", ".html")
UI_SEGMENTS = ("app", "components")
PROOF_KEY = "proof/{repo}#{number}@{sha}"

_ISSUE_RE = re.compile(r"\b(?:ENG|OPS|ENB)-\d+\b")


def is_ui(path: str) -> bool:
    """A UI file by extension, or any path segment named ``app`` or ``components``."""
    if path.endswith(UI_EXT):
        return True
    return any(segment in UI_SEGMENTS for segment in path.split("/"))


def proof_tenant(repo: str) -> str:
    return f"proof:{repo}"


def proof_body(pr: github_prs.OpenPr, key: str, repo: str) -> str:
    issue = key or "none: PR has no issue key, block"
    return f"""Follow skill visual-proof.

Repo: {repo}
PR: {pr.full_name}#{pr.number} ({pr.url})
Branch: {pr.branch}
Head sha: {pr.head_sha}
Issue: {issue}

Bring up the branch preview: `peak-preview up {repo} --ref {pr.branch} --json`.

Complete with metadata `{{"screenshots", "video", "comment", "head_sha"}}`; `head_sha` is {pr.head_sha}.
"""


async def reconcile_proofs(
    *,
    gh: str,
    org: str = "Peak-Digital-Software",
    board: str = "engineering",
    list_cards=kanban_cli.cards,
    create=kanban_cli.create,
    repos=github_prs.repos_by_production,
    prs=github_prs.open_prs,
    files=github_prs.changed_files,
) -> list[str]:
    """One pass; returns created card ids (at most one per production repo)."""
    created: list[str] = []
    try:
        production = await repos(gh, org)
    except Exception:
        logger.exception("pr proof: listing repos failed")
        return created
    for full_name, is_production in production.items():
        if not is_production:
            continue
        repo = full_name.rsplit("/", 1)[-1]
        tenant = proof_tenant(repo)
        try:
            existing = await list_cards(board, tenant)
            if any(is_open(card) for card in existing):
                continue
            known = {card.id for card in existing}
            for pr in sorted(await prs(gh, full_name), key=lambda p: p.number):
                if not any(is_ui(path) for path in await files(gh, full_name, pr.number)):
                    continue
                match = _ISSUE_RE.search(pr.title or "")
                card_id = await create(
                    board,
                    f"Visual proof: {repo}#{pr.number} @ {pr.head_sha[:7]}",
                    body=proof_body(pr, match.group(0) if match else "", repo),
                    assignee="visual-proof",
                    skills=["visual-proof"],
                    tenant=tenant,
                    idempotency_key=PROOF_KEY.format(repo=repo, number=pr.number, sha=pr.head_sha),
                    workspace="scratch",
                    completion_contract="local-only",
                )
                # The same key returns the existing card: this head was already proved.
                if card_id in known:
                    continue
                created.append(card_id)
                break
        except Exception:
            logger.exception("pr proof: %s failed", full_name)
    return created
