"""Tests for tool registry, executor, and agentic loop."""

import json
from unittest.mock import patch, MagicMock

import pytest


# TEST: All tool definitions have required fields
def test_tool_definitions_valid():
    from research.tools import TOOL_DEFINITIONS

    assert len(TOOL_DEFINITIONS) == 5
    for tool in TOOL_DEFINITIONS:
        assert tool["type"] == "function"
        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"


# TEST: execute_tool fetch_url calls fetch and returns content
def test_execute_tool_fetch_url():
    from research.tools import execute_tool

    with patch("research.fetch_http.fetch_url", return_value="# Hello World\nSome content here"):
        result = execute_tool("fetch_url", {"url": "https://example.com"})

    assert "Hello World" in result


# TEST: execute_tool fetch_hacker_news returns formatted stories
def test_execute_tool_hn():
    from research.tools import execute_tool

    fake_stories = [
        {"title": "Show HN: Cool Project", "score": 200, "url": "https://example.com/1",
         "comments_url": "https://news.ycombinator.com/item?id=1"},
        {"title": "GPT-5 Released", "score": 500, "url": "https://example.com/2",
         "comments_url": "https://news.ycombinator.com/item?id=2"},
    ]

    with patch("research.fetch_hn.fetch_hn_stories", return_value=fake_stories):
        result = execute_tool("fetch_hacker_news", {"count": 5})

    assert "Show HN: Cool Project" in result
    assert "GPT-5 Released" in result
    assert "200 pts" in result


# TEST: execute_tool with unknown tool returns error
def test_execute_tool_unknown():
    from research.tools import execute_tool

    result = execute_tool("nonexistent_tool", {})
    assert "unknown tool" in result.lower()


# TEST: execute_tool search_research_db queries and formats results
def test_execute_tool_search_db(tmp_path):
    from research.db import init_db, insert_source
    from research.tools import execute_tool

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    insert_source(db_path, {
        "url": "https://example.com/ai",
        "content_hash": "abc123",
        "source_type": "http",
        "fetched_at": "2026-03-02T00:00:00",
        "title": "AI Agents Overview",
        "summary": "A survey of recent AI agent frameworks.",
        "relevance_score": 0.9,
        "relevance_tags": '["AI agents"]',
        "key_developments": "[]",
    })

    result = execute_tool("search_research_db", {"days": 7}, db_path=db_path)
    assert "AI Agents Overview" in result
    assert "0.9" in result


# TEST: run_tool_loop returns direct reply when model uses no tools
def test_run_tool_loop_no_tools():
    from research.tools import run_tool_loop

    mock_response = {
        "role": "assistant",
        "content": "AI agents are software programs that act autonomously.",
        "tool_calls": None,
    }

    with patch("research.tools.ollama_chat", return_value=mock_response):
        result = run_tool_loop(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "what are AI agents?"}],
            caller="test",
        )

    assert result == "AI agents are software programs that act autonomously."


# TEST: run_tool_loop executes tool and returns final reply
def test_run_tool_loop_with_tool():
    from research.tools import run_tool_loop

    # First call: model requests a tool
    tool_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "fetch_hacker_news",
                "arguments": {"count": 5},
            }
        }],
    }
    # Second call: model gives final answer
    final_response = {
        "role": "assistant",
        "content": "Here are the top HN stories: GPT-5 was released.",
        "tool_calls": None,
    }

    fake_stories = [
        {"title": "GPT-5 Released", "score": 500, "url": "https://example.com",
         "comments_url": "https://news.ycombinator.com/item?id=1"},
    ]

    with patch("research.tools.ollama_chat", side_effect=[tool_response, final_response]), \
         patch("research.fetch_hn.fetch_hn_stories", return_value=fake_stories):
        result = run_tool_loop(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "what's on hacker news?"}],
            caller="test",
        )

    assert "GPT-5" in result


# TEST: run_tool_loop stops at max iterations
def test_run_tool_loop_max_iterations():
    from research.tools import run_tool_loop

    # Model keeps requesting tools forever
    tool_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "fetch_hacker_news",
                "arguments": {"count": 5},
            }
        }],
    }
    final_response = {
        "role": "assistant",
        "content": "Here's what I found after multiple attempts.",
        "tool_calls": None,
    }

    fake_stories = [
        {"title": "Story", "score": 100, "url": "https://example.com",
         "comments_url": "https://news.ycombinator.com/item?id=1"},
    ]

    # 3 tool responses + 1 final = 4 calls total
    with patch("research.tools.ollama_chat",
               side_effect=[tool_response, tool_response, tool_response, final_response]), \
         patch("research.fetch_hn.fetch_hn_stories", return_value=fake_stories):
        result = run_tool_loop(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "keep searching"}],
            caller="test",
            max_iterations=3,
        )

    assert result == "Here's what I found after multiple attempts."
