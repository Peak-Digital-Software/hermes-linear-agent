"""Persisted per-issue counter for agent-to-agent comment mentions."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class MentionCap:
    def __init__(self, path: Path, limit: int) -> None:
        self._path = Path(path)
        self._limit = limit
        self._lock = threading.Lock()

    def allow(self, issue_id: str) -> tuple[bool, int]:
        """Count one mention; ``(True, n)`` while ``n <= limit``, else ``(False, n)``."""
        with self._lock:
            counts = self._read()
            stored = counts.get(issue_id, 0)
            if stored > self._limit:
                return False, stored
            count = stored + 1
            counts[issue_id] = count
            self._write(counts)
            return count <= self._limit, count

    def _read(self) -> dict[str, int]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.warning("mention cap file %s unreadable; treating as empty", self._path)
            return {}
        if not isinstance(raw, dict):
            logger.warning("mention cap file %s malformed; treating as empty", self._path)
            return {}
        return {str(k): v for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)}

    def _write(self, counts: dict[str, int]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=self._path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(counts, handle)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
