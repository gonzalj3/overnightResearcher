"""Tests for user memory system (fact extraction, CRUD, profile building)."""

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def test_db(tmp_path):
    """Create a test database with all tables."""
    from research.db import init_db
    db_path = str(tmp_path / "test_memory.db")
    init_db(db_path)
    return db_path


# --- Mock LLM responses ---

def _mock_extract_add(text, existing_facts=None):
    """Mock that returns an ADD operation."""
    return [{"operation": "ADD", "fact_type": "interest", "fact_text": "interested in fusion energy", "confidence": 0.9}]


def _mock_extract_update(text, existing_facts=None):
    """Mock that returns an UPDATE operation."""
    if existing_facts:
        return [{"operation": "UPDATE", "fact_id": existing_facts[0]["id"],
                 "fact_type": "interest", "fact_text": "very interested in fusion energy", "confidence": 0.95}]
    return [{"operation": "NOOP"}]


def _mock_extract_delete(text, existing_facts=None):
    """Mock that returns a DELETE operation."""
    if existing_facts:
        return [{"operation": "DELETE", "fact_id": existing_facts[0]["id"]}]
    return [{"operation": "NOOP"}]


def _mock_extract_noop(text, existing_facts=None):
    """Mock that returns NOOP."""
    return [{"operation": "NOOP"}]


def _mock_extract_multi(text, existing_facts=None):
    """Mock that returns multiple operations."""
    return [
        {"operation": "ADD", "fact_type": "interest", "fact_text": "likes quantum computing", "confidence": 0.85},
        {"operation": "ADD", "fact_type": "project", "fact_text": "building a research agent", "confidence": 0.9},
    ]


# TEST UM.1: Save message and retrieve
def test_save_message(test_db):
    from research.db import insert_user_message, get_unprocessed_messages
    msg_id = insert_user_message(test_db, "What's new in fusion?", "+12104265298")
    assert msg_id > 0
    unprocessed = get_unprocessed_messages(test_db)
    assert len(unprocessed) == 1
    assert unprocessed[0]["message_text"] == "What's new in fusion?"


# TEST UM.2: Mark message as processed
def test_mark_processed(test_db):
    from research.db import insert_user_message, mark_message_processed, get_unprocessed_messages
    msg_id = insert_user_message(test_db, "Hello", "+12104265298")
    mark_message_processed(test_db, msg_id, extracted_facts=[{"operation": "NOOP"}])
    unprocessed = get_unprocessed_messages(test_db)
    assert len(unprocessed) == 0


# TEST UM.3: Insert and retrieve facts
def test_insert_and_get_facts(test_db):
    from research.db import insert_user_fact, get_active_facts
    insert_user_fact(test_db, "interest", "fusion energy", confidence=0.9)
    insert_user_fact(test_db, "project", "overnight research agent", confidence=0.85)
    facts = get_active_facts(test_db)
    assert len(facts) == 2
    assert facts[0]["fact_type"] in ("interest", "project")


# TEST UM.4: Update fact
def test_update_fact(test_db):
    from research.db import insert_user_fact, get_active_facts, update_fact
    fact_id = insert_user_fact(test_db, "interest", "fusion energy")
    update_fact(test_db, fact_id, fact_text="very interested in fusion energy", confidence=0.95)
    facts = get_active_facts(test_db)
    assert facts[0]["fact_text"] == "very interested in fusion energy"
    assert facts[0]["confidence"] == 0.95


# TEST UM.5: Delete fact (soft delete)
def test_delete_fact(test_db):
    from research.db import insert_user_fact, get_active_facts, update_fact
    fact_id = insert_user_fact(test_db, "dislike", "cryptocurrency")
    update_fact(test_db, fact_id, status="deleted")
    facts = get_active_facts(test_db)
    assert len(facts) == 0


# TEST UM.6: Find similar facts
def test_find_similar(test_db):
    from research.db import insert_user_fact, find_similar_facts
    insert_user_fact(test_db, "interest", "fusion energy research")
    similar = find_similar_facts(test_db, "fusion energy")
    assert len(similar) == 1


# TEST UM.7: Full process_message pipeline with ADD
def test_process_message_add(test_db):
    from research.user_memory import process_message
    with patch("research.user_memory.extract_facts_from_message", side_effect=_mock_extract_add):
        result = process_message(test_db, "I'm really into fusion energy research", "+12104265298")
    assert result["added"] == 1
    assert result["message_id"] > 0


# TEST UM.8: Process message with UPDATE
def test_process_message_update(test_db):
    from research.user_memory import process_message
    from research.db import insert_user_fact
    insert_user_fact(test_db, "interest", "interested in fusion energy")

    with patch("research.user_memory.extract_facts_from_message", side_effect=_mock_extract_update):
        result = process_message(test_db, "Actually I'm very into fusion", "+12104265298")
    assert result["updated"] == 1


# TEST UM.9: Build user profile stays under char limit
def test_user_profile_char_limit(test_db):
    from research.user_memory import build_user_profile
    from research.db import insert_user_fact
    # Insert many facts
    for i in range(30):
        insert_user_fact(test_db, "interest", f"Topic number {i} about some research area")
    profile = build_user_profile(test_db)
    assert len(profile) <= 800
    assert "<user_profile>" in profile
    assert "</user_profile>" in profile


# TEST UM.10: Build focus boost tags
def test_focus_boost_tags(test_db):
    from research.user_memory import build_focus_boost_tags
    from research.db import insert_user_fact
    insert_user_fact(test_db, "interest", "fusion energy")
    insert_user_fact(test_db, "project", "research agent")
    insert_user_fact(test_db, "preference", "concise reports")  # not included
    tags = build_focus_boost_tags(test_db)
    assert "fusion energy" in tags
    assert "research agent" in tags
    assert "concise reports" not in tags
