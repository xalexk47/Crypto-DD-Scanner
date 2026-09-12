"""Local persistence for the portfolio and the rotation engine.

Same shape as :mod:`src.history`: one SQLite file in the gitignored ``data/``
directory, no ORM, and the full object graph kept as a JSON payload so a future
version can render an old snapshot without a schema migration.

Four tables:

``wallets``             addresses you own, per chain
``position_meta``       your annotations on a holding: avg cost, ecosystem tag
``portfolio_snapshots`` one row per sync -- the equity curve lives here
``chain_heat``          one row per chain per refresh -- the heat history

Nothing here ever raises. Persistence is a convenience; a locked or corrupted
database must not take the dashboard down with it.
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
from .models import ChainHeat, CostBasisReport, PortfolioSnapshot, Wallet
from .utils import normalize_address, utcnow_iso

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    address   TEXT NOT NULL,
    chain     TEXT NOT NULL,
    label     TEXT DEFAULT '',
    added_at  TEXT NOT NULL,
    PRIMARY KEY (address, chain)
);

CREATE TABLE IF NOT EXISTS position_meta (
    chain         TEXT NOT NULL,
    address       TEXT NOT NULL,
    avg_cost_usd  REAL,
    tag           TEXT DEFAULT '',
    note          TEXT DEFAULT '',
    first_seen    TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (chain, address)
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at   TEXT NOT NULL,
    total_usd  REAL NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON portfolio_snapshots(taken_at DESC);

CREATE TABLE IF NOT EXISTS chain_heat (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    chain     TEXT NOT NULL,
    taken_at  TEXT NOT NULL,
    heat      REAL NOT NULL,
    state     TEXT NOT NULL,
    payload   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_heat_chain_time ON chain_heat(chain, taken_at DESC);

CREATE TABLE IF NOT EXISTS price_cache (
    chain       TEXT NOT NULL,
    address     TEXT NOT NULL,
    bucket      INTEGER NOT NULL,
    price_usd   REAL NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (chain, address, bucket)
);
"""

_lock = threading.Lock()


@contextmanager
def _connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or config.PORTFOLIO_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added after the first release. Existing databases are upgraded in
# place rather than recreated, so nobody loses the tags and costs they typed in.
_ADDED_COLUMNS = {
    "position_meta": {
        "basis_source": "TEXT DEFAULT ''",       # "manual" | "derived" | ""
        "basis_coverage_pct": "REAL",
        "realized_pnl_usd": "REAL",
        "derived_at": "TEXT",
        "basis_notes": "TEXT",
    },
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any column this version expects but an older database lacks."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, definition in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db(db_path: Optional[Path] = None) -> None:
    """Create the schema if it does not exist, and migrate it (idempotent)."""
    with _lock, _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


# --------------------------------------------------------------------------
# Wallets
# --------------------------------------------------------------------------
def add_wallet(address: str, chain: str, label: str = "", db_path: Optional[Path] = None) -> bool:
    """Register one wallet. Re-adding an existing one just updates the label."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO wallets (address, chain, label, added_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(address, chain) DO UPDATE SET label = excluded.label
                """,
                (normalize_address(address), chain, label or "", utcnow_iso()),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not add wallet: %s", exc)
        return False


def remove_wallet(address: str, chain: str, db_path: Optional[Path] = None) -> bool:
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            conn.execute(
                "DELETE FROM wallets WHERE address = ? AND chain = ?",
                (normalize_address(address), chain),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not remove wallet: %s", exc)
        return False


def list_wallets(chain: str = "", db_path: Optional[Path] = None) -> List[Wallet]:
    """Every registered wallet, optionally narrowed to one chain."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            if chain:
                rows = conn.execute(
                    "SELECT * FROM wallets WHERE chain = ? ORDER BY added_at", (chain,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM wallets ORDER BY chain, added_at").fetchall()
        return [
            Wallet(address=row["address"], chain=row["chain"], label=row["label"] or "",
                   added_at=row["added_at"])
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read wallets: %s", exc)
        return []


def replace_wallets(wallets: List[Wallet], db_path: Optional[Path] = None) -> int:
    """Swap the whole wallet list for a new one, in a single transaction.

    The UI edits wallets as one text area, so a partial write -- old rows gone,
    new ones not yet in -- would silently drop addresses on any failure.
    """
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            conn.execute("DELETE FROM wallets")
            conn.executemany(
                "INSERT INTO wallets (address, chain, label, added_at) VALUES (?, ?, ?, ?)",
                [
                    (normalize_address(w.address), w.chain, w.label or "", w.added_at or utcnow_iso())
                    for w in wallets
                ],
            )
        return len(wallets)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save wallets: %s", exc)
        return 0


# --------------------------------------------------------------------------
# Position metadata (avg cost, ecosystem tag, note)
# --------------------------------------------------------------------------
def set_position_meta(
    chain: str,
    address: str,
    avg_cost_usd: Optional[float] = None,
    tag: Optional[str] = None,
    note: Optional[str] = None,
    first_seen: Optional[str] = None,
    basis_source: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """Upsert your annotations on one holding.

    ``None`` means "leave as is" for tag and note, so the caller can update one
    field without reading the row first. Clearing a cost basis is done with the
    dedicated :func:`clear_avg_cost`, because ``None`` is also the value that
    represents "unknown basis" and the two must not be confused.
    """
    try:
        init_db(db_path)
        key = (chain, normalize_address(address))
        with _lock, _connect(db_path) as conn:
            existing = conn.execute(
                "SELECT * FROM position_meta WHERE chain = ? AND address = ?", key
            ).fetchone()
            conn.execute(
                """
                INSERT INTO position_meta
                    (chain, address, avg_cost_usd, tag, note, first_seen,
                     basis_source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chain, address) DO UPDATE SET
                    avg_cost_usd = excluded.avg_cost_usd,
                    tag          = excluded.tag,
                    note         = excluded.note,
                    first_seen   = excluded.first_seen,
                    basis_source = excluded.basis_source,
                    updated_at   = excluded.updated_at
                """,
                (
                    key[0], key[1],
                    avg_cost_usd if avg_cost_usd is not None
                    else (existing["avg_cost_usd"] if existing else None),
                    tag if tag is not None else ((existing["tag"] if existing else "") or ""),
                    note if note is not None else ((existing["note"] if existing else "") or ""),
                    first_seen or (existing["first_seen"] if existing else None) or utcnow_iso(),
                    basis_source if basis_source is not None
                    else ((existing["basis_source"] if existing else "") or ""),
                    utcnow_iso(),
                ),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save position metadata: %s", exc)
        return False


def clear_avg_cost(chain: str, address: str, db_path: Optional[Path] = None) -> bool:
    """Reset a holding back to "basis unknown"."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            conn.execute(
                "UPDATE position_meta SET avg_cost_usd = NULL, updated_at = ? "
                "WHERE chain = ? AND address = ?",
                (utcnow_iso(), chain, normalize_address(address)),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clear cost basis: %s", exc)
        return False


def all_position_meta(db_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Every annotation, keyed ``chain:address`` to match ``Position.key``."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            rows = conn.execute("SELECT * FROM position_meta").fetchall()
        return {f"{row['chain']}:{row['address']}": dict(row) for row in rows}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read position metadata: %s", exc)
        return {}


def save_derived_basis(report: "CostBasisReport", db_path: Optional[Path] = None) -> str:
    """Store a reconstructed basis, without ever clobbering one you typed in.

    Returns what happened: ``"saved"``, ``"kept_manual"`` (your own figure was
    left in place, realized P&L still recorded) or ``"failed"``.

    Realized P&L is written either way: it comes from the sells that actually
    happened, not from whichever average is on display.
    """
    try:
        init_db(db_path)
        key = (report.chain, normalize_address(report.token_address))
        with _lock, _connect(db_path) as conn:
            existing = conn.execute(
                "SELECT * FROM position_meta WHERE chain = ? AND address = ?", key
            ).fetchone()
            manual = bool(
                existing
                and (existing["basis_source"] or "") == "manual"
                and existing["avg_cost_usd"] is not None
            )
            notes = json.dumps(report.notes) if report.notes else None

            if manual:
                conn.execute(
                    """
                    UPDATE position_meta
                       SET realized_pnl_usd = ?, derived_at = ?, basis_notes = ?, updated_at = ?
                     WHERE chain = ? AND address = ?
                    """,
                    (report.realized_pnl_usd, report.derived_at or utcnow_iso(), notes,
                     utcnow_iso(), key[0], key[1]),
                )
                return "kept_manual"

            conn.execute(
                """
                INSERT INTO position_meta
                    (chain, address, avg_cost_usd, tag, note, first_seen, basis_source,
                     basis_coverage_pct, realized_pnl_usd, derived_at, basis_notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'derived', ?, ?, ?, ?, ?)
                ON CONFLICT(chain, address) DO UPDATE SET
                    avg_cost_usd       = excluded.avg_cost_usd,
                    basis_source       = 'derived',
                    basis_coverage_pct = excluded.basis_coverage_pct,
                    realized_pnl_usd   = excluded.realized_pnl_usd,
                    derived_at         = excluded.derived_at,
                    basis_notes        = excluded.basis_notes,
                    updated_at         = excluded.updated_at
                """,
                (
                    key[0], key[1], report.avg_cost_usd,
                    (existing["tag"] if existing else "") or "",
                    (existing["note"] if existing else "") or "",
                    (existing["first_seen"] if existing else None) or utcnow_iso(),
                    report.coverage_pct, report.realized_pnl_usd,
                    report.derived_at or utcnow_iso(), notes, utcnow_iso(),
                ),
            )
        return "saved"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save derived cost basis: %s", exc)
        return "failed"


# --------------------------------------------------------------------------
# Historical price cache
# --------------------------------------------------------------------------
# A price at a past timestamp cannot change, so these rows never expire. That
# makes re-deriving a basis almost free after the first run.
def get_cached_price(
    chain: str, address: str, bucket: int, db_path: Optional[Path] = None
) -> Optional[float]:
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            row = conn.execute(
                "SELECT price_usd FROM price_cache WHERE chain = ? AND address = ? AND bucket = ?",
                (chain, normalize_address(address), int(bucket)),
            ).fetchone()
        return float(row["price_usd"]) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read price cache: %s", exc)
        return None


def cache_prices(
    rows: List[Dict[str, Any]], db_path: Optional[Path] = None
) -> int:
    """Store ``{chain, address, bucket, price_usd}`` rows. Returns the count."""
    if not rows:
        return 0
    try:
        init_db(db_path)
        now = utcnow_iso()
        with _lock, _connect(db_path) as conn:
            conn.executemany(
                """
                INSERT INTO price_cache (chain, address, bucket, price_usd, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chain, address, bucket) DO UPDATE SET
                    price_usd = excluded.price_usd, fetched_at = excluded.fetched_at
                """,
                [
                    (row["chain"], normalize_address(row["address"]), int(row["bucket"]),
                     float(row["price_usd"]), now)
                    for row in rows
                ],
            )
        return len(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write price cache: %s", exc)
        return 0


# --------------------------------------------------------------------------
# Portfolio snapshots
# --------------------------------------------------------------------------
def record_snapshot(snapshot: PortfolioSnapshot, db_path: Optional[Path] = None) -> Optional[int]:
    """Persist one sync. This is what the equity curve is built from."""
    if not snapshot.positions and snapshot.total_usd <= 0:
        return None
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO portfolio_snapshots (taken_at, total_usd, payload) VALUES (?, ?, ?)",
                (
                    snapshot.taken_at or utcnow_iso(),
                    snapshot.total_usd,
                    json.dumps(snapshot.to_dict(), default=str),
                ),
            )
            return cursor.lastrowid
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not record portfolio snapshot: %s", exc)
        return None


def recent_snapshots(limit: int = 200, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Snapshot headers, newest first (payload excluded for speed)."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT id, taken_at, total_usd FROM portfolio_snapshots "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read portfolio snapshots: %s", exc)
        return []


def load_snapshot(snapshot_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    try:
        with _lock, _connect(db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM portfolio_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load snapshot %s: %s", snapshot_id, exc)
        return None


def previous_snapshot(db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The most recent stored snapshot, used as the baseline for deltas."""
    rows = recent_snapshots(limit=1, db_path=db_path)
    return load_snapshot(rows[0]["id"], db_path=db_path) if rows else None


def equity_curve(limit: int = 500, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Total value over time, oldest first, ready to plot."""
    rows = recent_snapshots(limit=limit, db_path=db_path)
    return list(reversed(rows))


# --------------------------------------------------------------------------
# Chain heat history
# --------------------------------------------------------------------------
def record_heat(heat: ChainHeat, db_path: Optional[Path] = None) -> Optional[int]:
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO chain_heat (chain, taken_at, heat, state, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    heat.chain,
                    heat.taken_at or utcnow_iso(),
                    heat.heat,
                    heat.state,
                    json.dumps(heat.to_dict(), default=str),
                ),
            )
            return cursor.lastrowid
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not record chain heat: %s", exc)
        return None


def heat_history(chain: str, limit: int = 200, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Heat readings for one chain, oldest first."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT id, chain, taken_at, heat, state FROM chain_heat "
                "WHERE chain = ? ORDER BY id DESC LIMIT ?",
                (chain, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read heat history: %s", exc)
        return []


def clear_portfolio(db_path: Optional[Path] = None) -> None:
    """Wipe snapshots and heat history, keeping wallets and annotations."""
    try:
        init_db(db_path)
        with _lock, _connect(db_path) as conn:
            conn.execute("DELETE FROM portfolio_snapshots")
            conn.execute("DELETE FROM chain_heat")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clear portfolio history: %s", exc)
