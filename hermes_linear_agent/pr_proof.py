"""Give every open PR head on a production repo one Verify card (one pass; the caller loops).

Kept apart from ``pr_reconcile`` so that module stays Oracle-only. Verifies are serialised
per repo (previews share ports): a repo with an open Verify card gets nothing new this
pass, and at most one new card per repo per pass. A PR whose title carries an issue key is
carded under that key so the status rules see it; otherwise under ``proof:<repo>``.
"""

from __future__ import annotations

import logging
import re

from . import github_prs, kanban_cli

logger = logging.getLogger(__name__)

UI_EXT = (".tsx", ".jsx", ".css", ".html")
UI_SEGMENTS = ("app", "components")
PROOF_KEY = "proof/{repo}#{number}@{sha}"
VERIFY_ASSIGNEE = "visual-proof"

_ISSUE_RE = re.compile(r"\b(?:ENG|OPS|ENB)-\d+\b")
# shortcut: named-vendor regex only, extend the tuple when a new client API appears.
_INTEGRATION_RE = re.compile(r"(?i)stripe|lightspeed|shopify|printnode")


def is_ui(path: str) -> bool:
    """A UI file by extension, or any path segment named ``app`` or ``components``."""
    if path.endswith(UI_EXT):
        return True
    return any(segment in UI_SEGMENTS for segment in path.split("/"))


def _touches_integration(path: str, patch: str) -> bool:
    if _INTEGRATION_RE.search(path):
        return True
    return any(
        line[:1] in "+-" and not line.startswith(("+++", "---")) and _INTEGRATION_RE.search(line)
        for line in patch.splitlines()
    )


def classify(files: list[tuple[str, str]]) -> list[str]:
    """Modes a PR needs, in order ``ui``, ``integration``; ``logic`` only when neither applies."""
    modes = []
    if any(is_ui(path) for path, _ in files):
        modes.append("ui")
    if any(_touches_integration(path, patch) for path, patch in files):
        modes.append("integration")
    return modes or ["logic"]


def proof_tenant(repo: str) -> str:
    return f"proof:{repo}"


def proof_body(pr: github_prs.OpenPr, key: str, repo: str, modes: list[str]) -> str:
    issue = key or "none: PR has no issue key, block"
    return f"""Follow skill visual-proof.

Mode: {", ".join(modes)}
Repo: {repo}
PR: {pr.full_name}#{pr.number} ({pr.url})
Branch: {pr.branch}
Head sha: {pr.head_sha}
Issue: {issue}

Bring up the branch preview: `peak-preview up {repo} --ref {pr.branch} --json`.

Complete with metadata `{{"evidence_url": <URL of the Linear comment carrying the evidence>, "head_sha": "{pr.head_sha}"}}`.
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
    files=github_prs.changed_patches,
    open_cards=kanban_cli.open_cards,
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
        try:
            prefix = f"Verify: {repo}#"
            if any(str(card.get("title", "")).startswith(prefix) for card in await open_cards(board, VERIFY_ASSIGNEE)):
                continue
            for pr in sorted(await prs(gh, full_name), key=lambda p: p.number):
                modes = classify(await files(gh, full_name, pr.number))
                match = _ISSUE_RE.search(pr.title or "")
                key = match.group(0) if match else ""
                tenant = key or proof_tenant(repo)
                known = {card.id for card in await list_cards(board, tenant)}
                card_id = await create(
                    board,
                    f"Verify: {repo}#{pr.number} @ {pr.head_sha[:7]}",
                    body=proof_body(pr, key, repo, modes),
                    assignee=VERIFY_ASSIGNEE,
                    skills=["visual-proof"],
                    tenant=tenant,
                    idempotency_key=PROOF_KEY.format(repo=repo, number=pr.number, sha=pr.head_sha),
                    workspace="scratch",
                    completion_contract="local-only",
                )
                # The same key returns the existing card: this head was already verified.
                if card_id in known:
                    continue
                created.append(card_id)
                break
        except Exception:
            logger.exception("pr proof: %s failed", full_name)
    return created
