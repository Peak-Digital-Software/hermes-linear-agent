"""Merge-gate: hold final replies that ask the human to merge too early.

A reply whose last paragraph asks the human to merge must not post as a
``response`` until the issue's Linear state is ``Ready to Merge`` (or
``Done``). Only the kanban sync sets Ready to Merge, after Verify. Earlier,
the reply posts as a ``thought``. Every hit is appended to
``merge-gate-hits.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from . import kanban_cli
from .go_gate import last_paragraph

logger = logging.getLogger(__name__)

MERGE_GATE_PATTERNS: tuple[str, ...] = (
    r"\bmerge\s+(?:it|#\d+|\[?[\w.-]+#\d+)",
    r"\bready\s+for\s+you\s+to\s+merge\b",
    r"\bnext:\s*merge\b",
)

_COMPILED = tuple((p, re.compile(p, re.IGNORECASE)) for p in MERGE_GATE_PATTERNS)

ALLOWED_STATE_NAMES = frozenset({"Ready to Merge", "Done"})
LOG_FILE = "merge-gate-hits.jsonl"


def match(text: str) -> str | None:
    """Return the source of the first pattern matching the last paragraph."""
    paragraph = last_paragraph(text)
    if not paragraph:
        return None
    for source, regex in _COMPILED:
        if regex.search(paragraph):
            return source
    return None


def allowed(issue: dict[str, Any] | None) -> bool:
    """True only when the issue's state is Ready to Merge or Done."""
    if not isinstance(issue, dict):
        return False
    state = issue.get("state") or {}
    return state.get("name") in ALLOWED_STATE_NAMES


def log_hit(
    *,
    issue: str | None,
    session: str | None,
    profile: str | None,
    pattern: str,
    posted_as: str,
    state: str | None,
) -> None:
    """Append one JSON line to ``merge-gate-hits.jsonl``; never raises."""
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "issue": issue,
        "session": session,
        "profile": profile,
        "pattern": pattern,
        "posted_as": posted_as,
        "state": state,
    }
    try:
        directory = kanban_cli._state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("[linear_agent] Could not write %s: %s", LOG_FILE, exc)
