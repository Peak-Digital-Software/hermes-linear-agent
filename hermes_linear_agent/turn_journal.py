"""Plugin-owned journal of Linear turns, so a gateway restart loses nothing."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JournalEntry:
    issue_id: str
    issue_key: str
    text: str
    at: float


def _entry(raw: Any) -> JournalEntry | None:
    if not isinstance(raw, dict):
        return None
    try:
        return JournalEntry(
            issue_id=str(raw["issue_id"]),
            issue_key=str(raw.get("issue_key", "")),
            text=str(raw.get("text", "")),
            at=float(raw.get("at", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


class TurnJournal:
    def __init__(self, path: Path, *, max_inflight_age_s: float = 3 * 3600) -> None:
        self._path = Path(path)
        self._max_age = max_inflight_age_s
        self._lock = threading.Lock()

    def queue(self, issue_id: str, issue_key: str, text: str) -> None:
        with self._lock:
            queued, inflight = self._read()
            queued.append(JournalEntry(issue_id, issue_key, text, time.time()))
            self._write(queued, inflight)

    def start(self, issue_id: str, issue_key: str, text: str) -> None:
        with self._lock:
            queued, inflight = self._read()
            inflight[issue_id] = JournalEntry(issue_id, issue_key, text, time.time())
            self._write(queued, inflight)

    def finish(self, issue_id: str) -> None:
        with self._lock:
            queued, inflight = self._read()
            if inflight.pop(issue_id, None) is not None:
                self._write(queued, inflight)

    def requeue(self, issue_id: str) -> bool:
        with self._lock:
            queued, inflight = self._read()
            entry = inflight.pop(issue_id, None)
            if entry is None:
                return False
            queued.append(entry)
            self._write(queued, inflight)
            return True

    def take_pending(self) -> tuple[list[JournalEntry], list[JournalEntry]]:
        with self._lock:
            queued, inflight = self._read()
            # shortcut: a turn that ends without a final send() stays in-flight; entries older than max_inflight_age_s are dropped instead of resumed — upgrade if core exposes turn-end reasons
            cutoff = time.time() - self._max_age
            interrupted = [e for e in inflight.values() if e.at >= cutoff]
            if queued or inflight:
                self._write([], {})
            return queued, interrupted

    def _read(self) -> tuple[list[JournalEntry], dict[str, JournalEntry]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("journal root is not an object")
            raw_queued = data.get("queued", [])
            raw_inflight = data.get("inflight", {})
            if not isinstance(raw_queued, list) or not isinstance(raw_inflight, dict):
                raise ValueError("journal has wrong shapes")
        except FileNotFoundError:
            return [], {}
        except (OSError, ValueError) as exc:
            logger.warning("turn journal %s unreadable, treating as empty: %s", self._path, exc)
            return [], {}
        queued = [e for e in map(_entry, raw_queued) if e is not None]
        inflight: dict[str, JournalEntry] = {}
        for key, raw in raw_inflight.items():
            entry = _entry(raw)
            if entry is not None:
                inflight[str(key)] = entry
        return queued, inflight

    def _write(self, queued: list[JournalEntry], inflight: dict[str, JournalEntry]) -> None:
        payload = {
            "queued": [asdict(e) for e in queued],
            "inflight": {k: asdict(v) for k, v in inflight.items()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=self._path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
