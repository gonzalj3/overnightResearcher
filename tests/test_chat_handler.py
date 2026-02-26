"""Tests for two-way iMessage chat handler."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def reports_dir(tmp_path):
    """Create a reports dir with a sample report."""
    rdir = tmp_path / "reports"
    rdir.mkdir()
    (rdir / "2026-02-25.md").write_text(
        "# AI Research Report — 2026-02-25\n\n"
        "## Executive Summary\n"
        "- AI agents are evolving rapidly\n"
        "- Local LLM inference is becoming more practical\n\n"
        "## Detailed Findings\n"
        "### Theme: AI Agents\n"
        "Several new frameworks were released this week."
    )
    return str(rdir)


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    from research.db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


# TEST: generate_reply builds correct prompt and returns LLM response
def test_generate_reply_prompt(test_db, reports_dir):
    from research.chat_handler import generate_reply

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "AI agents are evolving with new frameworks like CrewAI and AutoGen."
    }

    with patch("research.chat_handler.requests.post", return_value=mock_resp) as mock_post:
        reply = generate_reply(
            "What's new with AI agents?",
            db_path=test_db,
            reports_dir=reports_dir,
        )

    assert reply == "AI agents are evolving with new frameworks like CrewAI and AutoGen."

    # Verify prompt contains the report and question
    call_kwargs = mock_post.call_args[1]["json"]
    prompt = call_kwargs["prompt"]
    assert "AI Research Report" in prompt
    assert "What's new with AI agents?" in prompt
    assert call_kwargs["model"] == "qwen3:8b"


# TEST: handle_message skips own messages (is_from_me=true)
def test_handle_message_skips_own(test_db, reports_dir):
    from research.chat_handler import handle_message

    with patch("research.chat_handler.requests.post") as mock_ollama, \
         patch("research.chat_handler.subprocess.run") as mock_send:
        result = handle_message(
            {"text": "hello", "is_from_me": True, "sender": "+12104265298",
             "chat_id": 4},
            db_path=test_db,
            reports_dir=reports_dir,
        )

    assert result is False
    mock_ollama.assert_not_called()
    mock_send.assert_not_called()


# TEST: handle_message processes incoming and sends reply
def test_handle_message_incoming(test_db, reports_dir):
    from research.chat_handler import handle_message

    mock_ollama_resp = MagicMock()
    mock_ollama_resp.status_code = 200
    mock_ollama_resp.json.return_value = {
        "response": "Here's what I found about AI agents."
    }

    mock_send_result = MagicMock()
    mock_send_result.returncode = 0

    with patch("research.chat_handler.requests.post", return_value=mock_ollama_resp), \
         patch("research.chat_handler.subprocess.run", return_value=mock_send_result) as mock_send:
        result = handle_message(
            {"text": "Tell me about AI agents", "is_from_me": False, "sender": "+12104265298", "chat_id": 4},
            db_path=test_db,
            reports_dir=reports_dir,
        )

    assert result is True
    send_args = mock_send.call_args[0][0]
    assert send_args[0] == "imsg"
    assert send_args[1] == "send"
    assert "--to" in send_args
    assert "--text" in send_args
    assert "AI agents" in send_args[-1]


# TEST: handle_message sends fallback when Ollama is down
def test_handle_message_ollama_down(test_db, reports_dir):
    from research.chat_handler import handle_message
    import requests as req

    mock_send_result = MagicMock()
    mock_send_result.returncode = 0

    with patch("research.chat_handler.requests.post", side_effect=req.ConnectionError), \
         patch("research.chat_handler.subprocess.run", return_value=mock_send_result) as mock_send:
        result = handle_message(
            {"text": "What's new?", "is_from_me": False, "sender": "+12104265298", "chat_id": 4},
            db_path=test_db,
            reports_dir=reports_dir,
        )

    assert result is True
    reply_text = mock_send.call_args[0][0][-1]  # last arg is the --text value
    assert "can't answer right now" in reply_text.lower() or "ollama" in reply_text.lower()


# TEST: reply truncation at word boundary
def test_reply_truncation():
    from research.chat_handler import _truncate_reply

    long_text = "word " * 600  # 3000 chars
    truncated = _truncate_reply(long_text)
    assert len(truncated) <= 1504  # 1500 + "..."
    assert truncated.endswith("...")

    # Short text unchanged
    short = "Hello world"
    assert _truncate_reply(short) == "Hello world"
