"""GPU usage timesheet — tracks every Ollama inference call."""

import json
import logging
import os
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
TIMESHEET_PATH = os.path.join(
    os.path.dirname(__file__), "logs", "gpu-timesheet.jsonl"
)


def _append_entry(entry):
    """Append a timesheet entry to the JSONL log."""
    os.makedirs(os.path.dirname(TIMESHEET_PATH), exist_ok=True)
    try:
        with open(TIMESHEET_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to write timesheet entry: %s", e)


def ollama_generate(model, prompt, caller, system=None, use_json=False,
                    timeout=120, options=None):
    """Call Ollama /api/generate and log GPU usage to timesheet.

    Args:
        model: Ollama model name (e.g. "qwen3:32b")
        prompt: The prompt string
        caller: Identifier for what triggered this call (e.g. "summarizer",
                "synthesis.cluster", "chat_handler.reply")
        system: Optional system prompt
        use_json: If True, set format="json"
        timeout: Request timeout in seconds
        options: Dict of Ollama options (temperature, num_ctx, etc.)

    Returns:
        The raw response string from the model.

    Raises:
        requests.RequestException on HTTP errors.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if use_json:
        payload["format"] = "json"
    if options:
        payload["options"] = options

    start = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    error_msg = None

    try:
        resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "")
        return response_text
    except Exception as e:
        error_msg = str(e)
        raise
    finally:
        elapsed = time.monotonic() - start
        entry = {
            "start": start_utc,
            "duration_s": round(elapsed, 2),
            "model": model,
            "caller": caller,
            "prompt_len": len(prompt),
            "response_len": len(response_text) if error_msg is None else 0,
            "error": error_msg,
        }
        _append_entry(entry)


def ollama_chat(model, messages, caller, tools=None, timeout=120, options=None):
    """Call Ollama /api/chat with optional tool calling support.

    Args:
        model: Ollama model name (e.g. "qwen3.5:9b")
        messages: list of {role, content} dicts
        caller: Timesheet identifier
        tools: list of OpenAI-format tool definitions (optional)
        timeout: Request timeout in seconds
        options: Dict of Ollama options (temperature, num_ctx, etc.)

    Returns:
        dict: {"role": "assistant", "content": str|None, "tool_calls": list|None}
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if options:
        payload["options"] = options

    start = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    error_msg = None
    response_msg = None

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        response_msg = data.get("message", {})
        return response_msg
    except Exception as e:
        error_msg = str(e)
        raise
    finally:
        elapsed = time.monotonic() - start
        content_len = len(response_msg.get("content", "") or "") if response_msg else 0
        tool_calls = response_msg.get("tool_calls") if response_msg else None
        entry = {
            "start": start_utc,
            "duration_s": round(elapsed, 2),
            "model": model,
            "caller": caller,
            "prompt_len": sum(len(m.get("content", "") or "") for m in messages),
            "response_len": content_len,
            "tool_calls": len(tool_calls) if tool_calls else 0,
            "error": error_msg,
        }
        _append_entry(entry)
