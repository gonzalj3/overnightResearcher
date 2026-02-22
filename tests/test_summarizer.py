"""Chunk 3 tests: LLM Summarization Pipeline.

Tests 3.1, 3.2, 3.4, 3.5, 3.6, 3.8 require Ollama running with qwen3:32b.
Test 3.3 is pure Python (JSON repair).
Test 3.7 is pure Python (prompt budget).
"""

import json
import os
import shutil
import tempfile
import time
from datetime import date

import pytest


# TEST 3.1: Ollama API is reachable and responds to JSON format
def test_ollama_json_response():
    import requests
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen3:32b",
        "prompt": '/no_think\nRespond with ONLY this JSON: {"test": "hello"}',
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 2048},
    }, timeout=120)
    data = resp.json()
    parsed = json.loads(data["response"])
    assert "test" in parsed


# TEST 3.2: Summarizer produces valid structured output
def test_summarize_single_source():
    from research.summarizer import summarize_source
    content = """
    OpenAI released GPT-5 today with significant improvements in reasoning
    and code generation. The model shows 40% improvement on math benchmarks
    and introduces native tool use capabilities.
    """
    interests = ["AI agent frameworks", "local LLM inference"]
    result = summarize_source(content, interests, source_url="https://example.com")
    assert "title" in result
    assert "summary" in result
    assert "relevance_tags" in result
    assert "relevance_score" in result
    assert isinstance(result["relevance_score"], (int, float))
    assert 0 <= result["relevance_score"] <= 1
    assert isinstance(result["key_developments"], list)


# TEST 3.3: JSON repair handles common malformed outputs
def test_json_repair():
    from research.json_repair import repair_json
    # Trailing comma
    assert repair_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}
    # Markdown fences
    assert repair_json('```json\n{"a": 1}\n```') == {"a": 1}
    # Single quotes
    result = repair_json("{'a': 1}")
    assert result == {"a": 1}


# TEST 3.4: Summarizer handles empty/garbage content gracefully
def test_summarize_garbage():
    from research.summarizer import summarize_source
    result = summarize_source("", ["AI"], source_url="https://example.com")
    assert result is not None
    assert "summary" in result


# TEST 3.5: Batch summarizer processes manifest
def test_batch_summarizer():
    from research.summarizer import run_batch_summarize
    tmpdir = tempfile.mkdtemp()
    try:
        # Write test content files
        content_path_1 = os.path.join(tmpdir, "content_1.md")
        content_path_2 = os.path.join(tmpdir, "content_2.md")
        with open(content_path_1, "w") as f:
            f.write("Test article about AI agents and local inference on Apple Silicon.")
        with open(content_path_2, "w") as f:
            f.write("New research on transformer architectures for code generation.")

        test_manifest = {
            "date": date.today().isoformat(),
            "items": [
                {"url": "https://example.com/1", "content_hash": "abc123",
                 "content_path": content_path_1},
                {"url": "https://example.com/2", "content_hash": "def456",
                 "content_path": content_path_2},
            ],
        }

        results = run_batch_summarize(
            test_manifest,
            interests=["AI agents"],
            summaries_base_dir=os.path.join(tmpdir, "summaries"),
        )
        assert results["total_processed"] == 2
        assert results["total_succeeded"] >= 1
        assert os.path.isdir(results["summaries_dir"])
    finally:
        shutil.rmtree(tmpdir)


# TEST 3.6: Resume skips already-processed items
def test_batch_resume():
    from research.summarizer import run_batch_summarize
    tmpdir = tempfile.mkdtemp()
    try:
        content_path = os.path.join(tmpdir, "content_1.md")
        with open(content_path, "w") as f:
            f.write("Test article about AI.")

        test_manifest = {
            "date": date.today().isoformat(),
            "items": [
                {"url": "https://example.com/1", "content_hash": "resume_test_hash",
                 "content_path": content_path},
            ],
        }

        summaries_dir = os.path.join(tmpdir, "summaries")

        # Run once
        results1 = run_batch_summarize(
            test_manifest,
            interests=["AI"],
            summaries_base_dir=summaries_dir,
        )
        assert results1["total_processed"] == 1

        # Run again — should skip
        results2 = run_batch_summarize(
            test_manifest,
            interests=["AI"],
            summaries_base_dir=summaries_dir,
        )
        assert results2["total_skipped"] == 1
        assert results2["total_processed"] == 0
    finally:
        shutil.rmtree(tmpdir)


# TEST 3.7: Summarizer respects token budget
def test_summarizer_token_budget():
    from research.summarizer import build_prompt
    interests = ["AI agents", "local LLM"]
    content = "x " * 500  # ~500 tokens of content
    prompt = build_prompt(content, interests)
    # Rough token estimate: 1 token ≈ 4 chars
    estimated_tokens = len(prompt) / 4
    assert estimated_tokens < 4000, f"Prompt too large: ~{estimated_tokens} tokens"


# TEST 3.8: Performance — single summary under 60 seconds
def test_summary_performance():
    from research.summarizer import summarize_source
    content = "OpenAI announced a new model with improved reasoning capabilities. " * 20
    start = time.time()
    summarize_source(content, ["AI"], source_url="https://example.com")
    elapsed = time.time() - start
    assert elapsed < 60, f"Summary took {elapsed:.1f}s (max 60s)"
