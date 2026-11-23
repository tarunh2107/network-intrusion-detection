"""SQLite persistence layer for network intrusion prediction logs and security alerts.

Phase 7 Requirements:
- Parameterized SQL with ACID transactions, WAL mode, and appropriate indexes.
- Stores prediction ID, UTC timestamp, model version, threshold, score, and label.
- Does NOT store full raw 77-feature payloads by default.
- Preserves history across server restarts.
- Atomic batch inserts with automatic rollback on error.
- Note on Request Retries: Database transactions provide atomic writes, not a guaranteed
  at-least-once delivery mechanism. Retried prediction requests can create duplicate records.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    model_version TEXT NOT NULL,
    threshold REAL NOT NULL,
    attack_probability REAL NOT NULL,
    prediction_label TEXT NOT NULL
);
"""

CREATE_INDEX_TIMESTAMP = """
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions (timestamp_utc DESC);
"""

CREATE_INDEX_LABEL = """
CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions (prediction_label);
"""


def get_connection(db_path: pathlib.Path) -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database with WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_database(db_path: pathlib.Path) -> None:
    """Initialize database tables and indexes."""
    with get_connection(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.execute(CREATE_INDEX_TIMESTAMP)
        conn.execute(CREATE_INDEX_LABEL)
        conn.commit()
    logger.info("Database initialized successfully at %s", db_path)


def check_database_readiness(db_path: pathlib.Path) -> bool:
    """Test database connectivity and readiness."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute("SELECT 1;")
            row = cursor.fetchone()
            return bool(row and row[0] == 1)
    except Exception as exc:
        logger.error("Database readiness check failed on %s: %s", db_path, exc)
        return False


def record_prediction(
    db_path: pathlib.Path,
    prediction_id: str,
    timestamp_utc: str,
    model_version: str,
    threshold: float,
    attack_probability: float,
    prediction_label: str,
) -> None:
    """Persist a single prediction record atomically.

    Raises:
        sqlite3.Error: If transaction fails (rolled back automatically).
    """
    sql = """
    INSERT INTO predictions (
        id, timestamp_utc, model_version, threshold, attack_probability, prediction_label
    ) VALUES (?, ?, ?, ?, ?, ?);
    """
    with get_connection(db_path) as conn:
        conn.execute(
            sql,
            (
                prediction_id,
                timestamp_utc,
                model_version,
                float(threshold),
                float(attack_probability),
                prediction_label,
            ),
        )
        conn.commit()


def record_predictions_batch(
    db_path: pathlib.Path,
    records: List[Tuple[str, str, str, float, float, str]],
) -> None:
    """Persist multiple prediction records in a single atomic transaction.

    If any insertion fails, the entire batch rolls back.

    Args:
        db_path: Target SQLite database file.
        records: List of tuples (id, timestamp_utc, model_version, threshold, attack_probability, prediction_label).
    """
    if not records:
        return

    sql = """
    INSERT INTO predictions (
        id, timestamp_utc, model_version, threshold, attack_probability, prediction_label
    ) VALUES (?, ?, ?, ?, ?, ?);
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(sql, records)
    finally:
        conn.close()


def _normalize_iso_timestamp(ts_str: str) -> Tuple[datetime.datetime, str]:
    """Parse and normalize ISO timestamp string, handling URL-decoded '+' signs.

    Returns:
        Tuple of (datetime object, normalized ISO string).
    """
    s = ts_str.strip()
    # When '+' is sent in query strings without URL encoding (%2B), servers decode it to space.
    # Restore '+' when preceding a timezone offset like ' 00:00' or ' 05:30'.
    s = re.sub(r"(T|\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s+(\d{2}:\d{2})$", r"\1+\2", s)
    s = re.sub(r"[zZ]$", "+00:00", s)
    dt = datetime.datetime.fromisoformat(s)
    return dt, dt.isoformat()


def get_alerts(
    db_path: pathlib.Path,
    limit: int = 50,
    offset: int = 0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve stored ATTACK predictions with stable ordering, date filtering, and pagination.

    Args:
        db_path: Target SQLite database file.
        limit: Maximum alerts to return (1..500).
        offset: Pagination offset.
        start_time: Optional ISO timestamp filter.
        end_time: Optional ISO timestamp filter.

    Returns:
        Dictionary containing alerts list, total count, limit, and offset.

    Raises:
        ValueError: If date parameters are invalid or start_time > end_time.
    """
    if limit < 1 or limit > 500:
        raise ValueError("Pagination limit must be between 1 and 500.")
    if offset < 0:
        raise ValueError("Pagination offset must be non-negative.")

    # Validate timestamps if provided
    start_dt = None
    end_dt = None
    clean_start = None
    clean_end = None

    if start_time:
        try:
            start_dt, clean_start = _normalize_iso_timestamp(start_time)
        except Exception:
            raise ValueError(f"Invalid start_time ISO timestamp: {start_time!r}")
    if end_time:
        try:
            end_dt, clean_end = _normalize_iso_timestamp(end_time)
        except Exception:
            raise ValueError(f"Invalid end_time ISO timestamp: {end_time!r}")

    if start_dt and end_dt:
        cmp_start = start_dt if start_dt.tzinfo is not None else start_dt.replace(tzinfo=datetime.timezone.utc)
        cmp_end = end_dt if end_dt.tzinfo is not None else end_dt.replace(tzinfo=datetime.timezone.utc)
        if cmp_start > cmp_end:
            raise ValueError(f"start_time ({start_time}) cannot be later than end_time ({end_time}).")

    query_parts = ["WHERE prediction_label = 'ATTACK'"]
    params: List[Any] = []

    if clean_start:
        query_parts.append("AND timestamp_utc >= ?")
        params.append(clean_start)
    if clean_end:
        query_parts.append("AND timestamp_utc <= ?")
        params.append(clean_end)

    where_clause = " ".join(query_parts)

    count_sql = f"SELECT COUNT(*) FROM predictions {where_clause};"
    select_sql = f"""
    SELECT id, timestamp_utc, model_version, threshold, attack_probability, prediction_label
    FROM predictions
    {where_clause}
    ORDER BY timestamp_utc DESC, id DESC
    LIMIT ? OFFSET ?;
    """

    with get_connection(db_path) as conn:
        total_matching = conn.execute(count_sql, params).fetchone()[0]

        select_params = list(params) + [limit, offset]
        cursor = conn.execute(select_sql, select_params)
        rows = cursor.fetchall()

    alerts = [
        {
            "prediction_id": row["id"],
            "timestamp_utc": row["timestamp_utc"],
            "model_version": row["model_version"],
            "decision_threshold": row["threshold"],
            "attack_probability": row["attack_probability"],
            "prediction_label": row["prediction_label"],
        }
        for row in rows
    ]

    return {
        "total_alerts": total_matching,
        "limit": limit,
        "offset": offset,
        "alerts": alerts,
    }


def get_statistics(db_path: pathlib.Path) -> Dict[str, Any]:
    """Calculate aggregate statistics across stored prediction history.

    Handles empty database correctly.
    """
    sql = """
    SELECT
        COUNT(*) AS total_predictions,
        SUM(CASE WHEN prediction_label = 'BENIGN' THEN 1 ELSE 0 END) AS benign_predictions,
        SUM(CASE WHEN prediction_label = 'ATTACK' THEN 1 ELSE 0 END) AS attack_predictions
    FROM predictions;
    """
    with get_connection(db_path) as conn:
        row = conn.execute(sql).fetchone()

    total = row["total_predictions"] if row else 0
    benign = row["benign_predictions"] or 0 if row else 0
    attack = row["attack_predictions"] or 0 if row else 0

    proportion = (float(attack) / float(total)) if total > 0 else 0.0

    return {
        "total_predictions": total,
        "benign_predictions": benign,
        "attack_predictions": attack,
        "attack_proportion": round(proportion, 6),
        "disclaimer": (
            "These statistics reflect offline model classifications and do not "
            "represent ground-truth verified security incident counts."
        ),
    }
