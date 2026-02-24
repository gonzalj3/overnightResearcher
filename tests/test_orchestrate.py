"""Tests for Chunk 6: Orchestrator & Scheduling."""

import json
import os
import plistlib
import time
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    from research.db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


# TEST 6.1: Health check passes when everything is running
def test_health_check_passes():
    from research.orchestrate import run_health_check

    # Mock Ollama being available
    mock_tags_resp = MagicMock()
    mock_tags_resp.status_code = 200
    mock_tags_resp.json.return_value = {"models": [{"name": "qwen3:32b"}]}

    mock_gen_resp = MagicMock()
    mock_gen_resp.status_code = 200
    mock_gen_resp.json.return_value = {"response": "OK"}

    def mock_get(url, **kwargs):
        return mock_tags_resp

    def mock_post(url, **kwargs):
        return mock_gen_resp

    with patch("research.orchestrate.requests.get", side_effect=mock_get), \
         patch("research.orchestrate.requests.post", side_effect=mock_post):
        result = run_health_check()
        assert result["ollama_running"] is True
        assert result["ram_available_gb"] > 0
        assert result["disk_available_gb"] > 1
        assert result["model_responding"] is True


# TEST 6.2: Health check fails gracefully when Ollama is down
def test_health_check_ollama_down():
    from research.orchestrate import run_health_check

    import requests as req
    with patch("research.orchestrate.requests.get", side_effect=req.ConnectionError), \
         patch("research.orchestrate.time.sleep"):  # skip retry delays
        result = run_health_check()
        assert result["ollama_running"] is False
        assert result["model_responding"] is False


# TEST 6.3: Orchestrator runs end-to-end with mocked pipeline
def test_orchestrate_mini_run(tmp_path):
    from research.orchestrate import run_nightly_research

    # Create a minimal sources.yaml
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text("""
sources:
  hacker_news:
    type: hn
    max_items: 2
    enabled: true
interests:
  primary: [AI agents]
  secondary: []
output:
  raw_dir: {raw_dir}
""".format(raw_dir=str(tmp_path / "raw")))

    # Mock fetcher and summarizer to avoid network calls
    raw_dir = tmp_path / "raw" / "2026-02-22"
    raw_dir.mkdir(parents=True)

    # Write a fake raw file and manifest
    (raw_dir / "hn_0.md").write_text("# Test Article\nScore: 100\nContent about AI agents.")
    import hashlib
    content = "# Test Article\nScore: 100\nContent about AI agents."
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    manifest = [{"url": "https://example.com/1", "source_type": "hn",
                 "title": "Test Article", "fetched_at": "2026-02-22T00:00:00+00:00",
                 "content_hash": content_hash}]
    (raw_dir / "manifest.json").write_text(json.dumps(manifest))

    # Mock run_fetch_pipeline to return our pre-built dir
    mock_summary = {
        "title": "Test Article",
        "summary": "Article about AI agents.",
        "relevance_tags": ["AI agents"],
        "relevance_score": 0.8,
        "key_developments": ["New agent framework"],
        "source_url": "https://example.com/1",
        "content_hash": content_hash,
        "summarized_at": "2026-02-22T00:00:00+00:00",
    }
    summaries_dir = tmp_path / "summaries" / "2026-02-22"
    summaries_dir.mkdir(parents=True)
    (summaries_dir / f"{content_hash}.json").write_text(json.dumps(mock_summary))

    mock_summary_result = {
        "total_processed": 1,
        "total_succeeded": 1,
        "total_failed": 0,
        "total_skipped": 0,
        "summaries_dir": str(summaries_dir),
    }

    mock_synthesis = {
        "report_path": str(tmp_path / "reports" / "2026-02-22.md"),
        "themes_count": 1,
        "executive_summary": "AI agents continue to evolve.",
    }
    # Ensure report dir and file exist for the mock
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "2026-02-22.md").write_text(
        "# AI Research Report — 2026-02-22\n\n## Executive Summary\nTest."
    )

    with patch("research.orchestrate.run_fetch_pipeline", return_value=str(raw_dir)), \
         patch("research.orchestrate.run_batch_summarize", return_value=mock_summary_result), \
         patch("research.orchestrate.run_synthesis", return_value=mock_synthesis), \
         patch("research.orchestrate.send_imessage_notification", return_value=True) as mock_imsg:
        report_path = run_nightly_research(
            sources_override={"max_total": 5},
            db_path=str(tmp_path / "test.db"),
            output_dir=str(tmp_path / "reports"),
            log_dir=str(tmp_path / "logs"),
            sources_path=str(sources_yaml),
        )

    assert os.path.exists(report_path)
    with open(report_path) as f:
        content = f.read()
    assert "AI Research Report" in content

    # Verify iMessage notification was called with the report path
    mock_imsg.assert_called_once()
    call_args = mock_imsg.call_args
    assert call_args[0][0] == report_path  # first positional arg is report_path
    assert isinstance(call_args[0][1], str)  # second arg is executive_summary


# TEST 6.4: Orchestrator logs are created
def test_logging(tmp_path):
    from research.orchestrate import setup_logging

    log_dir = str(tmp_path / "logs")
    setup_logging(log_dir)

    import logging
    logger = logging.getLogger("research")
    logger.info("Test log message")

    log_files = os.listdir(log_dir)
    assert len(log_files) >= 1
    assert any(f.endswith(".log") for f in log_files)

    # Verify content was written
    log_path = os.path.join(log_dir, log_files[0])
    with open(log_path) as f:
        log_content = f.read()
    assert "Test log message" in log_content


# TEST 6.5: Orchestrator handles partial failure
def test_orchestrate_partial_failure(tmp_path):
    """If some fetches fail, pipeline should still produce a report."""
    from research.orchestrate import run_nightly_research

    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text("""
sources:
  hacker_news:
    type: hn
    max_items: 2
    enabled: true
interests:
  primary: [AI agents]
  secondary: []
output:
  raw_dir: {raw_dir}
""".format(raw_dir=str(tmp_path / "raw")))

    raw_dir = tmp_path / "raw" / "2026-02-22"
    raw_dir.mkdir(parents=True)
    # Empty manifest — simulates all fetches failed
    (raw_dir / "manifest.json").write_text("[]")

    mock_summary_result = {
        "total_processed": 0,
        "total_succeeded": 0,
        "total_failed": 0,
        "total_skipped": 0,
        "summaries_dir": str(tmp_path / "summaries" / "2026-02-22"),
    }

    mock_synthesis = {
        "report_path": str(tmp_path / "reports" / "2026-02-22.md"),
        "themes_count": 1,
        "executive_summary": "No data available.",
    }
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "2026-02-22.md").write_text(
        "# AI Research Report\n\nNo data."
    )

    with patch("research.orchestrate.run_fetch_pipeline", return_value=str(raw_dir)), \
         patch("research.orchestrate.run_batch_summarize", return_value=mock_summary_result), \
         patch("research.orchestrate.run_synthesis", return_value=mock_synthesis):
        report_path = run_nightly_research(
            sources_override={"max_total": 5},
            db_path=str(tmp_path / "test.db"),
            output_dir=str(tmp_path / "reports"),
            log_dir=str(tmp_path / "logs"),
            sources_path=str(sources_yaml),
        )

    assert os.path.exists(report_path)


# TEST 6.6: launchd plist is valid XML
def test_launchd_plist_valid(tmp_path):
    """Generate and validate a launchd plist."""
    plist_data = {
        "Label": "com.research.nightly",
        "ProgramArguments": [
            "/usr/bin/python3",
            os.path.join(os.path.dirname(__file__), "..", "research", "orchestrate.py"),
        ],
        "StartCalendarInterval": {
            "Hour": 0,
            "Minute": 0,
        },
        "StandardOutPath": "/tmp/research-nightly.log",
        "StandardErrorPath": "/tmp/research-nightly.err",
        "EnvironmentVariables": {
            "OLLAMA_HOST": "http://127.0.0.1:11434",
        },
    }
    plist_path = tmp_path / "com.research.nightly.plist"
    with open(plist_path, "wb") as f:
        plistlib.dump(plist_data, f)

    # Validate it can be re-read
    with open(plist_path, "rb") as f:
        loaded = plistlib.load(f)
    assert loaded["Label"] == "com.research.nightly"
    assert "ProgramArguments" in loaded
    assert "StartCalendarInterval" in loaded


# TEST 6.7: Manifest conversion for summarizer
def test_manifest_conversion(tmp_path):
    """Test that fetcher manifest is correctly converted for summarizer."""
    from research.orchestrate import _load_manifest_for_summarizer
    import hashlib

    raw_dir = tmp_path / "raw" / "2026-02-22"
    raw_dir.mkdir(parents=True)

    content = "# Test\nSome content here."
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    (raw_dir / "hn_0.md").write_text(content)
    manifest = [{"url": "https://example.com", "source_type": "hn",
                 "title": "Test", "fetched_at": "2026-02-22T00:00:00+00:00",
                 "content_hash": content_hash}]
    (raw_dir / "manifest.json").write_text(json.dumps(manifest))

    result = _load_manifest_for_summarizer(str(raw_dir))
    assert len(result["items"]) == 1
    assert result["items"][0]["content_hash"] == content_hash
    assert os.path.isfile(result["items"][0]["content_path"])


# TEST 6.8: iMessage notification sends successfully
def test_imessage_notification_success():
    from research.orchestrate import send_imessage_notification

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("research.orchestrate.subprocess.run", return_value=mock_result) as mock_run:
        result = send_imessage_notification(
            "/Users/jmg/reports/2026-02-24.md",
            "AI agents saw major advances this week."
        )

    assert result is True
    args = mock_run.call_args[0][0]
    assert "openclaw" in args
    assert "message" in args
    assert "send" in args
    assert "+12104265298" in args


# TEST 6.9: iMessage notification handles BlueBubbles 500 quirk
def test_imessage_notification_500_quirk():
    from research.orchestrate import send_imessage_notification

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error: BlueBubbles send failed (500): some error"

    with patch("research.orchestrate.subprocess.run", return_value=mock_result):
        result = send_imessage_notification("/Users/jmg/reports/2026-02-24.md")

    # Should still return True because BlueBubbles 500 often delivers
    assert result is True


# TEST 6.10: iMessage notification handles missing openclaw
def test_imessage_notification_no_openclaw():
    from research.orchestrate import send_imessage_notification

    with patch("research.orchestrate.subprocess.run", side_effect=FileNotFoundError):
        result = send_imessage_notification("/Users/jmg/reports/2026-02-24.md")

    assert result is False
