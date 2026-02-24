"""Tests for Chunk 7: Interest Profile Evolution & Multi-Day Continuity."""

import json
import os
from datetime import date, datetime, timedelta

import pytest


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    from research.db import init_db
    db_path = str(tmp_path / "test_research.db")
    init_db(db_path)
    return db_path


# TEST 7.1: Interest weights adjust based on relevance
def test_weight_adjustment(test_db):
    from research.interest_tracker import adjust_weights
    from research.db import update_interest_weights, get_interest_weights, insert_source

    # Seed some interests
    update_interest_weights(test_db, "AI agents", hits=5)
    update_interest_weights(test_db, "WebXR", hits=1)

    # Insert high-relevance sources tagged with "ai agents"
    for i in range(10):
        insert_source(test_db, {
            "url": f"https://example.com/adj_{i}",
            "content_hash": f"adj_hash_{i}",
            "source_type": "web",
            "fetched_at": datetime.now().isoformat(),
            "title": f"AI Agent Article {i}",
            "summary": "About AI agents.",
            "relevance_score": 0.85,
            "relevance_tags": json.dumps(["AI agents"]),
            "key_developments": json.dumps([]),
            "raw_content_path": f"/tmp/{i}.md",
        })

    adjusted = adjust_weights(test_db)
    weights = {w["interest"]: w["weight"]
               for w in get_interest_weights(test_db)}
    assert weights["AI agents"] > 1.0  # increased from high-relevance hits


# TEST 7.2: New interest suggestions from emerging tags
def test_suggest_new_interests(test_db):
    from research.interest_tracker import suggest_new_interests
    from research.db import insert_source

    # Insert sources with a tag NOT in the user's interests
    for i in range(5):
        insert_source(test_db, {
            "url": f"https://example.com/new_{i}",
            "content_hash": f"new_{i}",
            "source_type": "web",
            "fetched_at": datetime.now().isoformat(),
            "title": f"Quantum Computing {i}",
            "summary": "...",
            "relevance_score": 0.85,
            "relevance_tags": json.dumps(["quantum computing"]),
            "key_developments": json.dumps([]),
            "raw_content_path": f"/tmp/{i}.md",
        })

    suggestions = suggest_new_interests(
        test_db, current_interests=["AI agents"], min_occurrences=3
    )
    assert "quantum computing" in [s["tag"] for s in suggestions]


# TEST 7.3: Pinned interests are never reduced, blocked are removed
def test_pinned_and_blocked_interests():
    from research.interest_tracker import apply_overrides

    weights = {"AI agents": 2.0, "WebXR": 0.3, "blocked_topic": 1.0}
    overrides = {
        "pinned": ["AI agents"],
        "blocked": ["blocked_topic"],
        "focus": [],
    }
    result = apply_overrides(weights, overrides)
    assert result["AI agents"] >= 2.0  # never reduced
    assert "blocked_topic" not in result
    assert "WebXR" in result


# TEST 7.4: Weekly digest generates from 7 days of data
def test_weekly_digest(test_db):
    from research.interest_tracker import generate_weekly_digest
    from research.db import insert_source, insert_daily_report

    # Insert 7 days of data
    for day_offset in range(7):
        d = (date.today() - timedelta(days=day_offset)).isoformat()
        insert_source(test_db, {
            "url": f"https://example.com/weekly_{day_offset}",
            "content_hash": f"weekly_hash_{day_offset}",
            "source_type": "web",
            "fetched_at": (datetime.now() - timedelta(days=day_offset)).isoformat(),
            "title": f"Article from day {day_offset}",
            "summary": f"Summary {day_offset}",
            "relevance_score": 0.7 + day_offset * 0.03,
            "relevance_tags": json.dumps(["AI agents", "fusion"]),
            "key_developments": json.dumps([f"Dev from day {day_offset}"]),
            "raw_content_path": f"/tmp/w_{day_offset}.md",
        })
        insert_daily_report(test_db, {
            "report_date": d,
            "report_path": f"/tmp/reports/{d}.md",
            "total_sources": 50,
            "top_developments": json.dumps([f"Dev from day {day_offset}"]),
            "interests_snapshot": json.dumps(["AI agents", "fusion"]),
        })

    digest = generate_weekly_digest(test_db)
    assert "weekly" in digest.lower() or "Week" in digest
    assert len(digest) > 100
    assert "Total sources processed" in digest
