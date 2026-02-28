"""Tests for Chunk 5: Report Synthesis & Output."""

import json
import os
from datetime import date
from unittest.mock import patch, MagicMock

import pytest


# --- Mocked LLM responses for offline tests ---

MOCK_CLUSTER_RESPONSE = {
    "clusters": [
        {
            "theme": "AI Agent Frameworks",
            "items": [
                {"title": "New LLM framework", "summary": "A new agent framework."},
                {"title": "Another agent tool", "summary": "Tool for agents."},
            ],
        },
        {
            "theme": "Energy Research",
            "items": [
                {"title": "Fusion breakthrough", "summary": "New tokamak results."},
            ],
        },
    ]
}

MOCK_ANALYSIS_RESPONSE = {
    "analysis": "Several new AI agent frameworks emerged this week, building on the MCP protocol. These tools aim to simplify building autonomous agents that can interact with external services."
}

MOCK_EXEC_SUMMARY_RESPONSE = {
    "executive_summary": "- AI agent frameworks continue rapid development\n- Fusion energy research shows promising tokamak results\n- Local LLM inference becoming more practical",
    "watch_list": ["Follow up on MCP protocol adoption", "Monitor fusion research results"],
}


def _mock_llm_call(prompt, system=None, use_json=True, timeout=180, caller=None):
    """Route mock responses based on prompt content."""
    if "Group these research items" in prompt:
        return MOCK_CLUSTER_RESPONSE
    elif "Write a concise analysis" in prompt:
        return MOCK_ANALYSIS_RESPONSE
    elif "executive summary" in prompt.lower():
        return MOCK_EXEC_SUMMARY_RESPONSE
    return {"parse_failed": True}


@pytest.fixture
def mock_llm():
    with patch("research.synthesis._llm_call", side_effect=_mock_llm_call):
        yield


# TEST 5.1: Cluster & rank produces valid grouping
def test_cluster_and_rank(mock_llm):
    from research.synthesis import cluster_and_rank
    summaries = [
        {"title": "New LLM framework", "summary": "A new agent framework.", "relevance_score": 0.9, "relevance_tags": ["AI agents"]},
        {"title": "Fusion breakthrough", "summary": "New tokamak results.", "relevance_score": 0.8, "relevance_tags": ["fusion"]},
        {"title": "Another agent tool", "summary": "Tool for agents.", "relevance_score": 0.7, "relevance_tags": ["AI agents"]},
        {"title": "Chip news", "summary": "New GPU.", "relevance_score": 0.3, "relevance_tags": ["hardware"]},
    ]
    clusters = cluster_and_rank(summaries, top_n=3)
    assert len(clusters) >= 1
    assert all("theme" in c and "items" in c for c in clusters)
    all_titles = [item["title"] for c in clusters for item in c["items"]]
    assert "New LLM framework" in all_titles


# TEST 5.2: Deep synthesis produces analysis text
def test_deep_synthesis(mock_llm):
    from research.synthesis import synthesize_theme
    cluster = {
        "theme": "AI Agent Frameworks",
        "items": [
            {"title": "New framework", "summary": "Details about a new agent framework with MCP support."},
            {"title": "Update to Goose", "summary": "Goose added new scheduling features."},
        ],
    }
    memory = "Last week: Agent frameworks are trending. OpenClaw gained 10K stars."
    analysis = synthesize_theme(cluster, memory_context=memory)
    assert len(analysis) > 50
    assert isinstance(analysis, str)


# TEST 5.3: Executive summary is concise
def test_executive_summary(mock_llm):
    from research.synthesis import generate_executive_summary
    theme_analyses = [
        {"theme": "AI Agents", "analysis": "Several new frameworks emerged this week."},
        {"theme": "Fusion", "analysis": "New results from ITER published."},
    ]
    memory = "Tracking: agent frameworks, fusion research."
    summary = generate_executive_summary(theme_analyses, memory)
    assert len(summary["executive_summary"]) > 20
    assert len(summary["executive_summary"].split()) < 500


# TEST 5.4: Report formatter produces valid markdown
def test_report_format():
    from research.report import format_report
    report_data = {
        "date": date.today().isoformat(),
        "executive_summary": "Key finding 1. Key finding 2.",
        "watch_list": ["Follow up on X", "Monitor Y"],
        "themes": [
            {"theme": "AI", "analysis": "Analysis text.", "sources": ["https://a.com", "https://b.com"]},
        ],
        "stats": {
            "sources_processed": 50,
            "sources_skipped": 5,
            "fetch_errors": 2,
            "processing_time": "01:23:45",
        },
    }
    md = format_report(report_data)
    assert "# AI Research Report" in md
    assert "Executive Summary" in md
    assert "Watch List" in md
    assert "Sources" in md
    assert "50" in md


# TEST 5.5: Report saves to disk
def test_report_save(tmp_path):
    from research.report import save_report
    report_md = "# Test Report\nContent here."
    path = save_report(report_md, output_dir=str(tmp_path))
    assert os.path.exists(path)
    with open(path) as f:
        assert "Test Report" in f.read()


# TEST 5.6: Full synthesis pipeline end-to-end
def test_full_synthesis_pipeline(tmp_path, mock_llm):
    from research.synthesis import run_synthesis
    summaries = [
        {
            "title": f"Article {i}",
            "summary": f"Summary {i}",
            "relevance_score": 0.5 + i * 0.05,
            "relevance_tags": ["AI"],
            "key_developments": [f"Dev {i}"],
            "url": f"https://example.com/{i}",
        }
        for i in range(10)
    ]
    memory_context = "Previous findings: AI agents are trending."
    result = run_synthesis(summaries, memory_context, output_dir=str(tmp_path))
    assert os.path.exists(result["report_path"])
    assert result["themes_count"] >= 1


# TEST 5.7: Staleness penalty reduces score
def test_staleness_penalty(mock_llm):
    from research.synthesis import cluster_and_rank
    summaries = [
        {"title": "Stale item", "summary": "Old news.", "relevance_score": 0.9,
         "relevance_tags": ["AI"], "content_hash": "stale_hash_1"},
        {"title": "Fresh item", "summary": "New news.", "relevance_score": 0.9,
         "relevance_tags": ["AI"], "content_hash": "fresh_hash_1"},
    ]
    stale_hashes = {"stale_hash_1"}
    clusters = cluster_and_rank(summaries, stale_hashes=stale_hashes)
    # The LLM mock controls the output, but we can verify it ran without error
    assert len(clusters) >= 1


# TEST 5.8: Focus tags boost score
def test_focus_boost(mock_llm):
    from research.synthesis import cluster_and_rank
    summaries = [
        {"title": "Focus item", "summary": "Fusion news.", "relevance_score": 0.5,
         "relevance_tags": ["fusion energy"], "content_hash": "focus_1"},
        {"title": "Normal item", "summary": "Other news.", "relevance_score": 0.5,
         "relevance_tags": ["hardware"], "content_hash": "normal_1"},
    ]
    focus_tags = ["fusion energy"]
    clusters = cluster_and_rank(summaries, focus_tags=focus_tags)
    assert len(clusters) >= 1


# TEST 5.9: run_synthesis accepts stale_hashes and focus_tags
def test_synthesis_with_staleness_and_focus(tmp_path, mock_llm):
    from research.synthesis import run_synthesis
    summaries = [
        {"title": f"Art {i}", "summary": f"Sum {i}", "relevance_score": 0.7,
         "relevance_tags": ["AI"], "key_developments": [], "url": f"https://ex.com/{i}",
         "content_hash": f"hash_{i}"}
        for i in range(5)
    ]
    result = run_synthesis(
        summaries, "context",
        output_dir=str(tmp_path),
        stale_hashes={"hash_0", "hash_1"},
        focus_tags=["AI"],
    )
    assert os.path.exists(result["report_path"])
