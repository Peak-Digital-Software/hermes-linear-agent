"""GitHub helpers for the PR reconciler: internal repos, open PRs, local clones.

Everything shells out to a caller-supplied ``gh`` binary (never ``shell=True``),
with stdin closed, a timeout, and a ``RuntimeError`` carrying stderr on failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class OpenPr:
    repo: str
    full_name: str
    number: int
    url: str
    head_sha: str
    title: str
    base: str
    branch: str = ""


async def _run(gh: str, args: list[str]) -> str:
    """Run ``gh`` with ``args``; return trimmed stdout, raise with stderr."""
    proc = await asyncio.create_subprocess_exec(
        gh,
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        raise RuntimeError(f"gh {args[0] if args else ''} timed out after {TIMEOUT_SECONDS:.0f}s") from None
    if proc.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise RuntimeError(message or f"gh {' '.join(args)} exited {proc.returncode}")
    return stdout.decode(errors="replace").strip()


def _json_pages(text: str) -> list[Any]:
    """Parse one JSON document or several concatenated ones (``--paginate``)."""
    decoder = json.JSONDecoder()
    items: list[Any] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        value, pos = decoder.raw_decode(text, pos)
        if isinstance(value, list):
            items.extend(value)
        else:
            items.append(value)
    return items


async def _production_values(gh: str, org: str) -> dict[str, str | None]:
    """Full name -> raw ``production`` property value (None when unset); one ``gh api`` call."""
    out = await _run(gh, ["api", f"orgs/{org}/properties/values", "--paginate"])
    values: dict[str, str | None] = {}
    for row in _json_pages(out):
        if not isinstance(row, dict):
            continue
        full_name = row.get("repository_full_name")
        if not full_name:
            continue
        value = None
        for prop in row.get("properties") or []:
            if isinstance(prop, dict) and prop.get("property_name") == "production":
                value = prop.get("value")
                break
        values[full_name] = value
    return values


async def repos_by_production(gh: str, org: str) -> dict[str, bool]:
    """Full name -> whether ``production`` is ``"true"``; a repo with no value counts as False."""
    return {name: value == "true" for name, value in (await _production_values(gh, org)).items()}


async def internal_repos(gh: str, org: str) -> list[str]:
    """Full names whose ``production`` property is exactly ``"false"``."""
    return [name for name, value in (await _production_values(gh, org)).items() if value == "false"]


async def changed_files(gh: str, full_name: str, number: int) -> list[str]:
    """Paths touched by one PR."""
    out = await _run(gh, ["api", f"repos/{full_name}/pulls/{number}/files", "--paginate", "--jq", ".[].filename"])
    return [line for line in out.splitlines() if line.strip()]


async def open_prs(gh: str, full_name: str) -> list[OpenPr]:
    """Open, non-draft PRs of one repo."""
    out = await _run(
        gh,
        [
            "pr",
            "list",
            "-R",
            full_name,
            "--state",
            "open",
            "--json",
            "number,url,headRefOid,title,isDraft,baseRefName,headRefName",
        ],
    )
    repo = full_name.rsplit("/", 1)[-1]
    prs: list[OpenPr] = []
    for row in _json_pages(out) if out else []:
        if row.get("isDraft"):
            continue
        prs.append(
            OpenPr(
                repo=repo,
                full_name=full_name,
                number=int(row["number"]),
                url=row["url"],
                head_sha=row["headRefOid"],
                title=row["title"],
                base=row["baseRefName"],
                branch=row.get("headRefName") or "",
            )
        )
    return prs


async def ensure_clone(gh: str, full_name: str, root: Path) -> Path:
    """``root/<owner>/<repo>``; clones only when ``<path>/.git`` is missing."""
    owner, _, repo = full_name.partition("/")
    path = Path(root) / owner / repo
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        await _run(gh, ["repo", "clone", full_name, str(path)])
    return path
