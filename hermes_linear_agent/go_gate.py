"""Go-gate: detect final replies that end by asking the human for a go-ahead.

A reply whose last paragraph asks "reply go", "want me to", "should I" and the
like must not post as a ``response`` activity: that marks the Linear session
complete, so the human's answer starts a cold session. Such replies post as a
``thought`` instead. Every hit is appended to ``gate-hits.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from . import kanban_cli

logger = logging.getLogger(__name__)

GO_GATE_PATTERNS: tuple[str, ...] = (
    r"\breply\s+[\"'`]?go\b",
    r"\bsay\s+[\"'`]",
    r"\bfile it\?",
    r"\bwant me to\b",
    r"\bshall i\b",
    r"\bshould i\b",
)

_COMPILED = tuple((p, re.compile(p, re.IGNORECASE)) for p in GO_GATE_PATTERNS)
_BLANK_LINE = re.compile(r"\n\s*\n")

EXEMPT_STATE_NAME = "Ready to Merge"
LOG_FILE = "gate-hits.jsonl"


def last_paragraph(text: str) -> str:
    """Text after the last blank line, stripped."""
    return _BLANK_LINE.split((text or "").strip())[-1].strip()


def match(text: str) -> str | None:
    """Return the source of the first pattern matching the last paragraph."""
    paragraph = last_paragraph(text)
    if not paragraph:
        return None
    for source, regex in _COMPILED:
        if regex.search(paragraph):
            return source
    return None


def scope_for(issue: dict[str, Any] | None) -> str:
    """``gated`` or ``exempt`` for a fetched issue (``unknown`` when absent)."""
    if not isinstance(issue, dict):
        return "unknown"
    state = issue.get("state") or {}
    if state.get("name") == EXEMPT_STATE_NAME:
        return "exempt"
    if state.get("type") == "triage" and not issue.get("delegate"):
        return "exempt"
    return "gated"


def log_hit(
    *,
    issue: str | None,
    session: str | None,
    profile: str | None,
    pattern: str,
    posted_as: str,
    scope: str,
) -> None:
    """Append one JSON line to ``gate-hits.jsonl``; never raises."""
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "issue": issue,
        "session": session,
        "profile": profile,
        "pattern": pattern,
        "posted_as": posted_as,
        "scope": scope,
    }
    try:
        directory = kanban_cli._state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("[linear_agent] Could not write %s: %s", LOG_FILE, exc)
