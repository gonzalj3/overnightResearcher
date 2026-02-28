"""Tests for GPU timesheet tracking."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def timesheet_path(tmp_path):
    """Use a temporary timesheet file."""
    path = str(tmp_path / "gpu-timesheet.jsonl")
    with patch("research.gpu_timesheet.TIMESHEET_PATH", path):
        yield path


def test_ollama_generate_logs_entry(timesheet_path):
    """Successful call writes a timesheet entry with correct fields."""
    from research.gpu_timesheet import ollama_generate

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Hello world"}

    with patch("research.gpu_timesheet.requests.post", return_value=mock_resp):
        result = ollama_generate(
            model="qwen3:8b",
            prompt="Say hello",
            caller="test",
        )

    assert result == "Hello world"

    with open(timesheet_path) as f:
        entries = [json.loads(line) for line in f]

    assert len(entries) == 1
    e = entries[0]
    assert e["model"] == "qwen3:8b"
    assert e["caller"] == "test"
    assert e["prompt_len"] == len("Say hello")
    assert e["response_len"] == len("Hello world")
    assert e["error"] is None
    assert e["duration_s"] >= 0
    assert "start" in e


def test_ollama_generate_logs_error(timesheet_path):
    """Failed call still writes a timesheet entry with the error."""
    from research.gpu_timesheet import ollama_generate
    import requests as req

    with patch("research.gpu_timesheet.requests.post", side_effect=req.ConnectionError("refused")):
        with pytest.raises(req.ConnectionError):
            ollama_generate(
                model="qwen3:32b",
                prompt="test prompt",
                caller="test.error",
            )

    with open(timesheet_path) as f:
        entries = [json.loads(line) for line in f]

    assert len(entries) == 1
    e = entries[0]
    assert e["model"] == "qwen3:32b"
    assert e["caller"] == "test.error"
    assert e["error"] is not None
    assert "refused" in e["error"]
    assert e["response_len"] == 0


def test_multiple_calls_append(timesheet_path):
    """Multiple calls append to the same file."""
    from research.gpu_timesheet import ollama_generate

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "ok"}

    with patch("research.gpu_timesheet.requests.post", return_value=mock_resp):
        ollama_generate(model="qwen3:8b", prompt="one", caller="batch.1")
        ollama_generate(model="qwen3:32b", prompt="two", caller="batch.2")
        ollama_generate(model="qwen3:8b", prompt="three", caller="batch.3")

    with open(timesheet_path) as f:
        entries = [json.loads(line) for line in f]

    assert len(entries) == 3
    assert [e["caller"] for e in entries] == ["batch.1", "batch.2", "batch.3"]
    assert [e["model"] for e in entries] == ["qwen3:8b", "qwen3:32b", "qwen3:8b"]


def test_passes_options_and_json_format(timesheet_path):
    """Wrapper correctly passes system, format, and options to Ollama."""
    from research.gpu_timesheet import ollama_generate

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "{}"}

    with patch("research.gpu_timesheet.requests.post", return_value=mock_resp) as mock_post:
        ollama_generate(
            model="qwen3:8b",
            prompt="test",
            caller="test.options",
            system="Be concise.",
            use_json=True,
            timeout=30,
            options={"temperature": 0.1, "num_ctx": 4096},
        )

    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["model"] == "qwen3:8b"
    assert payload["system"] == "Be concise."
    assert payload["format"] == "json"
    assert payload["options"] == {"temperature": 0.1, "num_ctx": 4096}
    assert call_kwargs[1]["timeout"] == 30
