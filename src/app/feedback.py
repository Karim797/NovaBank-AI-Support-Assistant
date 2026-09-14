"""Prediction log + user feedback store.

Decision -> Reason -> Alternative -> Trade-off
- Decision: SQLite via the stdlib `sqlite3` module for the MVP, behind a thin
  `FeedbackStore` class.
- Reason: it is the only persistence choice with zero infrastructure, and the
  write volume of a demo is trivially served by it. The class boundary is what
  matters: nothing outside this file knows the backend.
- Alternative: PostgreSQL from day one.
- Trade-off: SQLite is a single file, so it does not survive a container restart
  unless mounted as a volume, does not support more than one writer process, and
  cannot be queried from a BI tool. The moment the service runs more than one
  replica, this must become Postgres — that is a `DATABASE_URL` change plus
  swapping the driver in this file, and the schema below is written to port
  cleanly (no SQLite-specific types).

Why log predictions at all: `request_id` is the join key between "what the model
said" and "what actually happened". Ground-truth labels for a support assistant
arrive late (a thumbs-down, an escalation, an agent's correction). Without a row
written at prediction time there is nothing to join them to, and drift
monitoring has no reference window.

PII: the raw question is **not** stored. We store its length and a SHA-256
fingerprint so repeated questions can be counted and a specific complaint can be
matched, without holding customer text in a demo database.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id        TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    question_hash     TEXT NOT NULL,
    question_chars    INTEGER NOT NULL,
    intent            TEXT NOT NULL,
    intent_confidence REAL NOT NULL,
    route             TEXT NOT NULL,
    source_ids        TEXT NOT NULL,
    retrieval_top_score REAL NOT NULL,
    answer_chars      INTEGER NOT NULL,
    needs_escalation  INTEGER NOT NULL,
    model_version     TEXT NOT NULL,
    llm_model         TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    latency_ms        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    helpful     INTEGER NOT NULL,
    comment     TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_intent ON predictions(intent);
"""


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


class FeedbackStore:
    def __init__(self, db_url: str) -> None:
        self.path = self._path_from_url(db_url)
        self._lock = threading.Lock()  # sqlite3 allows one writer at a time
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @staticmethod
    def _path_from_url(db_url: str) -> str:
        if db_url.startswith("sqlite:///"):
            return db_url[len("sqlite:///") :]
        if db_url.startswith("sqlite://"):
            return db_url[len("sqlite://") :]
        raise ValueError(
            f"unsupported FEEDBACK_DB_URL {db_url!r}; the MVP store is SQLite only"
        )

    def log_prediction(self, **row) -> None:
        cols = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO predictions (created_at, {cols}) "
                f"VALUES (?, {placeholders})",
                (_now(), *row.values()),
            )
            self._conn.commit()

    def log_feedback(self, request_id: str, helpful: bool, comment: str | None) -> bool:
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM predictions WHERE request_id = ?", (request_id,)
            ).fetchone()
            self._conn.execute(
                "INSERT OR REPLACE INTO feedback (request_id, created_at, helpful, comment) "
                "VALUES (?, ?, ?, ?)",
                (request_id, _now(), int(helpful), comment),
            )
            self._conn.commit()
        return bool(exists)

    def recent_predictions(self, limit: int = 1000) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def helpfulness_rate(self) -> tuple[int, float | None]:
        row = self._conn.execute(
            "SELECT COUNT(*), AVG(helpful) FROM feedback"
        ).fetchone()
        return int(row[0]), (float(row[1]) if row[1] is not None else None)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def ensure_parent(path: str) -> None:
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
