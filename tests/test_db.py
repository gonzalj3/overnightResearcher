"""Tests for Chunk 4: Knowledge Base & Persistent Memory."""

import json
import os
import sqlite3
from datetime import date, datetime, timedelta

import pytest


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    from research.db import init_db
    db_path = str(tmp_path / "test_research.db")
    init_db(db_path)
    return db_path


# TEST 4.1: Database initializes with correct schema
def test_db_init(test_db):
    conn = sqlite3.connect(test_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "sources" in table_names
    assert "daily_reports" in table_names
    assert "interest_feedback" in table_names
    conn.close()


# TEST 4.2: Insert and retrieve source
def test_insert_source(test_db):
    from research.db import insert_source, get_recent_sources
    source = {
        "url": "https://example.com/article",
        "content_hash": "abc123",
        "source_type": "web",
        "fetched_at": datetime.now().isoformat(),
        "title": "Test Article",
        "summary": "This is a test.",
        "relevance_score": 0.8,
        "relevance_tags": json.dumps(["AI", "agents"]),
        "key_developments": json.dumps(["New framework released"]),
        "raw_content_path": "/tmp/raw/abc123.md",
    }
    insert_source(test_db, source)
    recent = get_recent_sources(test_db, days=1)
    assert len(recent) == 1
    assert recent[0]["title"] == "Test Article"


# TEST 4.3: Deduplication works
def test_dedup(test_db):
    from research.db import insert_source, get_recent_sources
    source = {
        "url": "https://example.com/article",
        "content_hash": "same_hash",
        "source_type": "web",
        "fetched_at": datetime.now().isoformat(),
        "title": "Test",
        "summary": "Test",
        "relevance_score": 0.5,
        "relevance_tags": "[]",
        "key_developments": "[]",
        "raw_content_path": "/tmp/raw/same.md",
    }
    insert_source(test_db, source)
    insert_source(test_db, source)  # duplicate
    recent = get_recent_sources(test_db, days=1)
    assert len(recent) == 1  # not 2


# TEST 4.4: Interest weights update correctly
def test_interest_weights(test_db):
    from research.db import update_interest_weights, get_interest_weights
    update_interest_weights(test_db, "AI agents", hits=5)
    update_interest_weights(test_db, "AI agents", hits=3)
    weights = get_interest_weights(test_db)
    ai_weight = next(w for w in weights if w["interest"] == "AI agents")
    assert ai_weight["total_hits"] == 8


# TEST 4.5: Trending topics aggregation
def test_trending_topics(test_db):
    from research.db import insert_source, get_trending_topics
    for i in range(5):
        insert_source(test_db, {
            "url": f"https://example.com/{i}",
            "content_hash": f"hash_{i}",
            "source_type": "web",
            "fetched_at": datetime.now().isoformat(),
            "title": f"Article {i}",
            "summary": "test",
            "relevance_score": 0.7,
            "relevance_tags": json.dumps(["AI agents", "local LLM"]),
            "key_developments": json.dumps([]),
            "raw_content_path": f"/tmp/{i}.md",
        })
    topics = get_trending_topics(test_db, days=7)
    assert "AI agents" in [t["tag"] for t in topics]


# TEST 4.6: Memory context builder stays within token budget
def test_memory_context_budget(test_db):
    from research.memory import build_memory_context
    from research.db import insert_source
    for i in range(20):
        insert_source(test_db, {
            "url": f"https://example.com/{i}",
            "content_hash": f"hash_mem_{i}",
            "source_type": "web",
            "fetched_at": (datetime.now() - timedelta(days=i % 7)).isoformat(),
            "title": f"Historical Article {i}",
            "summary": f"Summary of article {i} about AI developments.",
            "relevance_score": 0.5 + (i % 5) * 0.1,
            "relevance_tags": json.dumps(["AI"]),
            "key_developments": json.dumps([f"Development {i}"]),
            "raw_content_path": f"/tmp/{i}.md",
        })
    context = build_memory_context(test_db)
    estimated_tokens = len(context) / 4
    assert estimated_tokens < 1000, f"Memory context too large: ~{estimated_tokens} tokens"
    assert len(context) > 0


# TEST 4.7: is_already_processed works for dedup
def test_already_processed(test_db):
    from research.db import insert_source, is_already_processed
    insert_source(test_db, {
        "url": "https://example.com/1",
        "content_hash": "existing_hash",
        "source_type": "web",
        "fetched_at": datetime.now().isoformat(),
        "title": "t",
        "summary": "s",
        "relevance_score": 0.5,
        "relevance_tags": "[]",
        "key_developments": "[]",
        "raw_content_path": "/tmp/x.md",
    })
    assert is_already_processed(test_db, "existing_hash") is True
    assert is_already_processed(test_db, "new_hash") is False


# TEST 4.8: mark_sources_reported and get_stale_hashes
def test_staleness(test_db):
    from research.db import insert_source, mark_sources_reported, get_stale_hashes
    insert_source(test_db, {
        "url": "https://example.com/stale",
        "content_hash": "stale_hash",
        "source_type": "web",
        "fetched_at": datetime.now().isoformat(),
        "title": "Stale",
        "summary": "Old news",
        "relevance_score": 0.7,
        "relevance_tags": "[]",
        "key_developments": "[]",
        "raw_content_path": "/tmp/stale.md",
    })
    # Mark as reported 5 days ago
    old_date = (date.today() - timedelta(days=5)).isoformat()
    mark_sources_reported(test_db, ["stale_hash"], report_date=old_date)
    stale = get_stale_hashes(test_db, days=3)
    assert "stale_hash" in stale


# TEST 4.9: get_stale_hashes excludes recent reports
def test_staleness_recent(test_db):
    from research.db import insert_source, mark_sources_reported, get_stale_hashes
    insert_source(test_db, {
        "url": "https://example.com/fresh",
        "content_hash": "fresh_hash",
        "source_type": "web",
        "fetched_at": datetime.now().isoformat(),
        "title": "Fresh",
        "summary": "New",
        "relevance_score": 0.9,
        "relevance_tags": "[]",
        "key_developments": "[]",
        "raw_content_path": "/tmp/fresh.md",
    })
    mark_sources_reported(test_db, ["fresh_hash"], report_date=date.today().isoformat())
    stale = get_stale_hashes(test_db, days=3)
    assert "fresh_hash" not in stale


# TEST 4.10: get_effective_weights with decay
def test_effective_weights_decay(test_db):
    from research.db import update_interest_weights, get_effective_weights
    import sqlite3
    update_interest_weights(test_db, "old topic", hits=10)
    # Manually set last_seen_at to 7 days ago to test decay
    conn = sqlite3.connect(test_db)
    old_time = (datetime.now() - timedelta(days=7)).isoformat()
    conn.execute("UPDATE interest_feedback SET last_seen_at = ? WHERE interest = ?",
                 (old_time, "old topic"))
    conn.commit()
    conn.close()
    weights = get_effective_weights(test_db)
    old_w = next(w for w in weights if w["interest"] == "old topic")
    # stored weight = 1.0 + (10 * 0.1) = 2.0
    # effective should be < stored due to 7 days of decay
    assert old_w["stored_weight"] == 2.0
    assert old_w["weight"] < old_w["stored_weight"]


# TEST 4.11: get_effective_weights without decay (recent)
def test_effective_weights_no_decay(test_db):
    from research.db import update_interest_weights, get_effective_weights
    update_interest_weights(test_db, "fresh topic", hits=5)
    weights = get_effective_weights(test_db)
    fresh_w = next(w for w in weights if w["interest"] == "fresh topic")
    # Just set, so decay should be negligible
    assert fresh_w["weight"] > 1.4  # stored = 1.5, decay minimal


# TEST 4.12: user_messages and user_memory tables exist
def test_new_tables_exist(test_db):
    conn = sqlite3.connect(test_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "user_messages" in table_names
    assert "user_memory" in table_names
    conn.close()
