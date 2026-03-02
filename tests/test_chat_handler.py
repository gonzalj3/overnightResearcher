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


# --- Fetch detection tests ---

def test_detect_fetch_explicit_url():
    from research.chat_handler import detect_fetch_request

    result = detect_fetch_request("grab https://simonwillison.net/2026/Feb/25/some-post")
    assert result["should_fetch"] is True
    assert result["url"] == "https://simonwillison.net/2026/Feb/25/some-post"
    assert "site" not in result
    assert "source" not in result


def test_detect_fetch_site_name():
    from research.chat_handler import detect_fetch_request

    result = detect_fetch_request("pull the latest from simonwillison.net")
    assert result["should_fetch"] is True
    assert result["site"] == "simonwillison.net"
    assert "url" not in result


def test_detect_fetch_hn():
    from research.chat_handler import detect_fetch_request

    result = detect_fetch_request("what's on hacker news?")
    assert result["should_fetch"] is True
    assert result["source"] == "hn"
    assert result["topic"] is None

    # With topic
    result2 = detect_fetch_request("what's on HN about OpenAI?")
    assert result2["should_fetch"] is True
    assert result2["source"] == "hn"
    assert result2["topic"] == "OpenAI"


def test_detect_fetch_no_match():
    from research.chat_handler import detect_fetch_request

    result = detect_fetch_request("tell me about AI agents")
    assert result["should_fetch"] is False
    assert "url" not in result
    assert "site" not in result
    assert "source" not in result


def test_handle_fetch_url(test_db):
    from research.chat_handler import handle_fetch_request

    mock_ollama_resp = MagicMock()
    mock_ollama_resp.status_code = 200
    mock_ollama_resp.json.return_value = {
        "response": "Simon wrote about new SQLite features and datasette updates."
    }

    with patch("research.fetch_http.requests.get") as mock_get, \
         patch("research.chat_handler.requests.post", return_value=mock_ollama_resp):
        # Mock the HTTP fetch
        mock_http_resp = MagicMock()
        mock_http_resp.status_code = 200
        mock_http_resp.text = "<html><body><h1>Simon's Blog</h1><p>New SQLite features</p></body></html>"
        mock_http_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_http_resp

        fetch_info = {
            "should_fetch": True,
            "url": "https://simonwillison.net",
            "query": "what's the latest on simonwillison.net",
        }
        reply = handle_fetch_request(fetch_info, db_path=test_db)

    assert "SQLite" in reply or "sqlite" in reply.lower() or "datasette" in reply.lower()


def test_handle_fetch_hn_no_topic(test_db):
    from research.chat_handler import handle_fetch_request

    fake_stories = [
        {"title": "Show HN: My Project", "url": "https://example.com/1", "score": 200,
         "comments_url": "https://news.ycombinator.com/item?id=1"},
        {"title": "GPT-5 Released", "url": "https://example.com/2", "score": 500,
         "comments_url": "https://news.ycombinator.com/item?id=2"},
        {"title": "Rust 2.0", "url": "https://example.com/3", "score": 300,
         "comments_url": "https://news.ycombinator.com/item?id=3"},
    ]

    with patch("research.fetch_hn.requests.get") as mock_get:
        # Mock topstories
        mock_top = MagicMock()
        mock_top.json.return_value = [1, 2, 3]
        mock_top.raise_for_status = MagicMock()

        # Mock individual items
        def side_effect(url, timeout=10):
            for i, story in enumerate(fake_stories, 1):
                if str(i) in url and "item" in url:
                    m = MagicMock()
                    m.json.return_value = {
                        "title": story["title"], "url": story["url"],
                        "score": story["score"], "id": i,
                    }
                    m.raise_for_status = MagicMock()
                    return m
            return mock_top

        mock_get.side_effect = side_effect

        fetch_info = {"should_fetch": True, "source": "hn", "topic": None, "query": "what's on HN?"}
        reply = handle_fetch_request(fetch_info, db_path=test_db)

    assert "GPT-5 Released" in reply  # highest score should be first
    assert "500 pts" in reply
    # No LLM call for no-topic case
    assert "Top Hacker News stories" in reply


def test_handle_fetch_hn_with_topic(test_db):
    from research.chat_handler import handle_fetch_request

    fake_stories = [
        {"title": "OpenAI releases GPT-5", "url": "https://example.com/1", "score": 400,
         "comments_url": "https://news.ycombinator.com/item?id=1"},
        {"title": "Rust 2.0 announced", "url": "https://example.com/2", "score": 300,
         "comments_url": "https://news.ycombinator.com/item?id=2"},
        {"title": "OpenAI safety report", "url": "https://example.com/3", "score": 200,
         "comments_url": "https://news.ycombinator.com/item?id=3"},
    ]

    with patch("research.fetch_hn.requests.get") as mock_hn_get:
        # Mock HN API
        mock_top = MagicMock()
        mock_top.json.return_value = [1, 2, 3]
        mock_top.raise_for_status = MagicMock()

        def hn_side_effect(url, timeout=10):
            for i, story in enumerate(fake_stories, 1):
                if str(i) in url and "item" in url:
                    m = MagicMock()
                    m.json.return_value = {
                        "title": story["title"], "url": story["url"],
                        "score": story["score"], "id": i,
                    }
                    m.raise_for_status = MagicMock()
                    return m
            return mock_top

        mock_hn_get.side_effect = hn_side_effect

        # Mock LLM filtering
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.status_code = 200
        mock_ollama_resp.json.return_value = {
            "response": '["OpenAI releases GPT-5", "OpenAI safety report"]'
        }

        with patch("research.chat_handler.requests.post", return_value=mock_ollama_resp):
            fetch_info = {"should_fetch": True, "source": "hn", "topic": "OpenAI",
                          "query": "HN posts about OpenAI"}
            reply = handle_fetch_request(fetch_info, db_path=test_db)

    assert "OpenAI" in reply
    assert "Rust" not in reply  # Should be filtered out
    assert 'about "OpenAI"' in reply


# --- Tool-calling path tests ---

# TEST: generate_reply_with_tools returns reply when model answers directly
def test_generate_reply_with_tools_direct(test_db, reports_dir):
    from research.chat_handler import generate_reply_with_tools

    mock_response = {
        "role": "assistant",
        "content": "AI agents are autonomous programs that can take actions.",
        "tool_calls": None,
    }

    with patch("research.tools.ollama_chat", return_value=mock_response):
        reply = generate_reply_with_tools(
            "what are AI agents?",
            db_path=test_db,
            reports_dir=reports_dir,
        )

    assert reply == "AI agents are autonomous programs that can take actions."


# TEST: generate_reply_with_tools returns None on failure, enabling fallback
def test_generate_reply_with_tools_failure(test_db, reports_dir):
    from research.chat_handler import generate_reply_with_tools
    import requests as req

    with patch("research.tools.ollama_chat", side_effect=req.ConnectionError("Ollama down")):
        reply = generate_reply_with_tools(
            "what's new?",
            db_path=test_db,
            reports_dir=reports_dir,
        )

    assert reply is None


# TEST: handle_message falls back to manual routing when tool calling fails
def test_handle_message_tool_fallback(test_db, reports_dir):
    from research.chat_handler import handle_message
    import requests as req

    # Tool calling fails (Ollama connection error)
    # Fallback generate_reply also needs to be mocked
    mock_ollama_resp = MagicMock()
    mock_ollama_resp.status_code = 200
    mock_ollama_resp.json.return_value = {
        "response": "Here is the fallback reply about AI."
    }

    mock_send_result = MagicMock()
    mock_send_result.returncode = 0

    with patch("research.tools.ollama_chat", side_effect=req.ConnectionError), \
         patch("research.chat_handler.requests.post", return_value=mock_ollama_resp), \
         patch("research.chat_handler.subprocess.run", return_value=mock_send_result) as mock_send:
        result = handle_message(
            {"text": "tell me about AI", "is_from_me": False, "sender": "+12104265298"},
            db_path=test_db,
            reports_dir=reports_dir,
        )

    assert result is True
    reply_text = mock_send.call_args[0][0][-1]
    assert "fallback reply" in reply_text.lower() or "AI" in reply_text
