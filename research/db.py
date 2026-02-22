"""SQLite database manager for persistent research memory."""

import json
import os
import sqlite3
from datetime import date, datetime, timedelta


SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "research.db")


def _connect(db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create tables if they don't exist."""
    conn = _connect(db_path)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.close()


def insert_source(db_path, source_data):
    """Insert a source with dedup (IGNORE on hash conflict)."""
    conn = _connect(db_path)
    conn.execute(
        """INSERT OR IGNORE INTO sources
           (url, content_hash, source_type, fetched_at, title, summary,
            relevance_score, relevance_tags, key_developments, raw_content_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            source_data["url"],
            source_data["content_hash"],
            source_data["source_type"],
            source_data["fetched_at"],
            source_data.get("title", ""),
            source_data.get("summary", ""),
            source_data.get("relevance_score", 0.0),
            source_data.get("relevance_tags", "[]"),
            source_data.get("key_developments", "[]"),
            source_data.get("raw_content_path", ""),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_sources(db_path, days=7):
    """Get sources from the last N days."""
    conn = _connect(db_path)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM sources WHERE fetched_at >= ? ORDER BY fetched_at DESC",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_top_developments(db_path, days=7, limit=20):
    """Get highest-relevance items from the last N days."""
    conn = _connect(db_path)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT * FROM sources
           WHERE fetched_at >= ? AND relevance_score IS NOT NULL
           ORDER BY relevance_score DESC LIMIT ?""",
        (cutoff, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_already_processed(db_path, content_hash):
    """Check if a content hash already exists in the DB."""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT 1 FROM sources WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    conn.close()
    return row is not None


def update_interest_weights(db_path, interest, hits=1):
    """Increment hit count and weight for an interest."""
    conn = _connect(db_path)
    existing = conn.execute(
        "SELECT id, total_hits, weight FROM interest_feedback WHERE interest = ?",
        (interest,),
    ).fetchone()
    today = date.today().isoformat()
    if existing:
        new_hits = existing["total_hits"] + hits
        new_weight = 1.0 + (new_hits * 0.1)
        conn.execute(
            """UPDATE interest_feedback
               SET total_hits = ?, weight = ?, last_updated = ?
               WHERE id = ?""",
            (new_hits, new_weight, today, existing["id"]),
        )
    else:
        conn.execute(
            """INSERT INTO interest_feedback (interest, weight, last_updated, total_hits)
               VALUES (?, ?, ?, ?)""",
            (interest, 1.0 + (hits * 0.1), today, hits),
        )
    conn.commit()
    conn.close()


def get_interest_weights(db_path):
    """Get all interest weights."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT interest, weight, total_hits, last_updated FROM interest_feedback ORDER BY weight DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_daily_report(db_path, report_data):
    """Log report metadata."""
    conn = _connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO daily_reports
           (report_date, report_path, total_sources, top_developments, interests_snapshot)
           VALUES (?, ?, ?, ?, ?)""",
        (
            report_data["report_date"],
            report_data["report_path"],
            report_data.get("total_sources", 0),
            report_data.get("top_developments", "[]"),
            report_data.get("interests_snapshot", "{}"),
        ),
    )
    conn.commit()
    conn.close()


def get_trending_topics(db_path, days=14):
    """Aggregate relevance tags by frequency over the last N days."""
    conn = _connect(db_path)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT relevance_tags FROM sources WHERE fetched_at >= ?",
        (cutoff,),
    ).fetchall()
    conn.close()

    tag_counts = {}
    for row in rows:
        tags_str = row["relevance_tags"]
        if not tags_str:
            continue
        try:
            tags = json.loads(tags_str)
        except (json.JSONDecodeError, TypeError):
            continue
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return sorted(
        [{"tag": t, "count": c} for t, c in tag_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )
