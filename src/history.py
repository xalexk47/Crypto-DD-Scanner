"""Local analysis history, stored in SQLite.

Kept intentionally tiny: one table, one file, no ORM.  The full result is
persisted as JSON so future versions can render an old report without a
schema migration, while the indexed columns support fast listing and search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import config
from .models import AnalysisResult
from .utils import normalize_address, utcnow_iso

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    address      TEXT NOT NULL,
    chain        TEXT NOT NULL,
    symbol       TEXT,
    name         TEXT,
    composite    REAL,
    decision     TEXT,
    market_cap   REAL,
    liquidity    REAL,
    price_usd    REAL,
    analyzed_at  TEXT NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_addr ON analyses(address, chain);
CREATE INDEX IF NOT EXISTS idx_analyses_time ON analyses(analyzed_at DESC);
"""

_lock = threading.Lock()


@contextmanager
def _connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or config.HISTORY_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Create the schema if it does not exist (idempotent)."""
    with _lock, _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def record(result: AnalysisResult, db_path: Optional[Path] = None) -> Optional[int]:
    """Persist one analysis.  Never raises - history is a nice-to-have."""
    if not result.ok or not result.snapshot:
        return None
    try:
        init_db(db_path)
        snapshot = result.snapshot
        with _lock, _connect(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses
                    (address, chain, symbol, name, composite, decision,
                     market_cap, liquidity, price_usd, analyzed_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalize_address(result.address),
                    result.chain,
                    snapshot.symbol,
                    snapshot.name,
                    result.composite,
                    result.decision,
                    snapshot.market_cap,
                    snapshot.liquidity_usd,
                    snapshot.price_usd,
                    result.analyzed_at or utcnow_iso(),
                    json.dumps(result.to_dict(), default=str),
                ),
            )
            return cursor.lastrowid
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write analysis history: %s", exc)
        return None


def recent(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Most recent analyses, newest first (payload excluded for speed)."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, address, chain, symbol, name, composite, decision,
                       market_cap, liquidity, price_usd, analyzed_at
                FROM analyses ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read analysis history: %s", exc)
        return []


def load(analysis_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load one stored analysis payload by row id."""
    try:
        with _lock, _connect(db_path) as conn:
            row = conn.execute("SELECT payload FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return json.loads(row["payload"]) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load analysis %s: %s", analysis_id, exc)
        return None


def clear(db_path: Optional[Path] = None) -> None:
    """Wipe all stored history."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            conn.execute("DELETE FROM analyses")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clear history: %s", exc)
