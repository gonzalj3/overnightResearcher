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
    """Create tables if they don't exist, then run migrations."""
    conn = _connect(db_path)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    # Migrations: add columns to existing tables (idempotent)
    migrations = [
        "ALTER TABLE sources ADD COLUMN last_reported_date DATE",
        "ALTER TABLE sources ADD COLUMN first_seen_date DATE",
        "ALTER TABLE interest_feedback ADD COLUMN last_seen_at DATETIME",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
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
    now = datetime.now().isoformat()
    if existing:
        new_hits = existing["total_hits"] + hits
        new_weight = 1.0 + (new_hits * 0.1)
        conn.execute(
            """UPDATE interest_feedback
               SET total_hits = ?, weight = ?, last_updated = ?, last_seen_at = ?
               WHERE id = ?""",
            (new_hits, new_weight, today, now, existing["id"]),
        )
    else:
        conn.execute(
            """INSERT INTO interest_feedback (interest, weight, last_updated, total_hits, last_seen_at)
               VALUES (?, ?, ?, ?, ?)""",
            (interest, 1.0 + (hits * 0.1), today, hits, now),
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


def mark_sources_reported(db_path, content_hashes, report_date=None):
    """Mark sources as reported on a given date."""
    if not content_hashes:
        return
    report_date = report_date or date.today().isoformat()
    conn = _connect(db_path)
    for h in content_hashes:
        conn.execute(
            "UPDATE sources SET last_reported_date = ? WHERE content_hash = ?",
            (report_date, h),
        )
    conn.commit()
    conn.close()


def get_stale_hashes(db_path, days=3):
    """Get content hashes last reported 3+ days ago (should be deprioritized)."""
    conn = _connect(db_path)
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT content_hash FROM sources WHERE last_reported_date IS NOT NULL AND last_reported_date <= ?",
        (cutoff,),
    ).fetchall()
    conn.close()
    return {row["content_hash"] for row in rows}


def get_effective_weights(db_path, decay_rate=0.995):
    """Get interest weights with time-based decay applied at read-time.

    Decay formula: effective_weight = stored_weight * (decay_rate ^ hours_since_last_seen)
    Half-life at 0.995: ~138 hours (~5.75 days)
    """
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT interest, weight, total_hits, last_updated, last_seen_at FROM interest_feedback ORDER BY weight DESC"
    ).fetchall()
    conn.close()

    now = datetime.now()
    results = []
    for r in rows:
        stored_weight = r["weight"]
        last_seen = r["last_seen_at"]
        if last_seen:
            try:
                last_dt = datetime.fromisoformat(last_seen)
                hours_elapsed = (now - last_dt).total_seconds() / 3600
                effective = stored_weight * (decay_rate ** hours_elapsed)
            except (ValueError, TypeError):
                effective = stored_weight
        else:
            effective = stored_weight
        results.append({
            "interest": r["interest"],
            "weight": round(effective, 3),
            "stored_weight": stored_weight,
            "total_hits": r["total_hits"],
            "last_updated": r["last_updated"],
            "last_seen_at": last_seen,
        })

    results.sort(key=lambda x: x["weight"], reverse=True)
    return results


# --- User message and memory functions ---

def insert_user_message(db_path, message_text, sender):
    """Store an incoming user message."""
    conn = _connect(db_path)
    cursor = conn.execute(
        "INSERT INTO user_messages (message_text, sender) VALUES (?, ?)",
        (message_text, sender),
    )
    msg_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def get_unprocessed_messages(db_path):
    """Get messages not yet processed for fact extraction."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM user_messages WHERE processed = 0 ORDER BY received_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_message_processed(db_path, message_id, extracted_facts=None):
    """Mark a message as processed, optionally storing extracted facts."""
    conn = _connect(db_path)
    conn.execute(
        "UPDATE user_messages SET processed = 1, extracted_facts = ? WHERE id = ?",
        (json.dumps(extracted_facts) if extracted_facts else None, message_id),
    )
    conn.commit()
    conn.close()


def insert_user_fact(db_path, fact_type, fact_text, confidence=0.8, source_message_id=None):
    """Insert a new user fact into the memory table."""
    conn = _connect(db_path)
    now = datetime.now().isoformat()
    cursor = conn.execute(
        """INSERT INTO user_memory (fact_type, fact_text, confidence, source_message_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (fact_type, fact_text, confidence, source_message_id, now, now),
    )
    fact_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return fact_id


def get_active_facts(db_path):
    """Get all active user facts."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM user_memory WHERE status = 'active' ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_fact(db_path, fact_id, fact_text=None, confidence=None, status=None):
    """Update an existing user fact."""
    conn = _connect(db_path)
    updates = []
    params = []
    if fact_text is not None:
        updates.append("fact_text = ?")
        params.append(fact_text)
    if confidence is not None:
        updates.append("confidence = ?")
        params.append(confidence)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if updates:
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(fact_id)
        conn.execute(
            f"UPDATE user_memory SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
    conn.close()


def find_similar_facts(db_path, fact_text, fact_type=None):
    """Find existing facts with similar text (substring match)."""
    conn = _connect(db_path)
    # Use LIKE for simple substring matching
    query = "SELECT * FROM user_memory WHERE status = 'active' AND fact_text LIKE ?"
    params = [f"%{fact_text[:50]}%"]
    if fact_type:
        query += " AND fact_type = ?"
        params.append(fact_type)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
