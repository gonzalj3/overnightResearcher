"""Two-way iMessage chat handler via imsg watch."""

import json
import logging
import os
import subprocess
import time
from datetime import date

import requests

from research.memory import build_memory_context

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
IMESSAGE_TARGET = "+12104265298"
PHONE_CHAT_ID = 4   # chat with phone number identifier — agent sends replies here
EMAIL_CHAT_ID = 5    # chat with email identifier — user sends questions from phone here
MAX_REPLY_CHARS = 1500


def _truncate_reply(text, max_chars=MAX_REPLY_CHARS):
    """Truncate text at word boundary."""
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def _get_latest_report(reports_dir):
    """Read today's report, falling back to most recent."""
    reports_dir = os.path.expanduser(reports_dir)
    if not os.path.isdir(reports_dir):
        return ""

    # Try today first
    today_path = os.path.join(reports_dir, f"{date.today().isoformat()}.md")
    if os.path.isfile(today_path):
        with open(today_path) as f:
            return f.read()

    # Fall back to most recent .md file
    md_files = sorted(
        [f for f in os.listdir(reports_dir) if f.endswith(".md")],
        reverse=True,
    )
    if md_files:
        with open(os.path.join(reports_dir, md_files[0])) as f:
            return f.read()

    return ""


def generate_reply(question, db_path="research/research.db", reports_dir="~/reports"):
    """Generate a reply to a user question using Ollama 8B.

    Reads today's report and DB memory context, then asks the LLM.

    Returns reply string.
    """
    report_text = _get_latest_report(reports_dir)
    memory_context = ""
    try:
        memory_context = build_memory_context(db_path)
    except Exception as e:
        logger.warning("Could not load memory context: %s", e)

    # Build user profile for personalized replies
    user_profile = ""
    try:
        from research.user_memory import build_user_profile
        user_profile = build_user_profile(db_path)
    except Exception as e:
        logger.debug("Could not load user profile: %s", e)

    # Truncate report to leave room for prompt + response
    report_excerpt = report_text[:4000] if report_text else "No report available today."

    prompt = f"""/no_think
You are a helpful AI research assistant. Answer the user's question based on the research report and memory context below. Keep your answer concise (2-3 short paragraphs) and iMessage-friendly.

{user_profile}

=== Today's Research Report ===
{report_excerpt}

=== Historical Context ===
{memory_context if memory_context else "No historical context available."}

=== User Question ===
{question}"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 8192},
            },
            timeout=120,
        )
        resp.raise_for_status()
        reply = resp.json().get("response", "").strip()
        return _truncate_reply(reply)
    except requests.ConnectionError:
        logger.error("Ollama is not running — cannot generate reply")
        report_path = os.path.join(
            os.path.expanduser(reports_dir),
            f"{date.today().isoformat()}.md",
        )
        return f"I can't answer right now (Ollama is down). Check the report at {report_path}"
    except Exception as e:
        logger.error("Error generating reply: %s", e)
        return "Sorry, I encountered an error generating a reply. Please try again later."


def handle_message(message_dict, db_path="research/research.db", reports_dir="~/reports"):
    """Process an incoming message and send a reply.

    Args:
        message_dict: Parsed JSON from imsg watch (has text, is_from_me, sender, etc.)
        db_path: Path to SQLite database.
        reports_dir: Path to reports directory.

    Returns:
        True if reply was sent, False otherwise.
    """
    # Skip messages sent from this Mac (agent replies).
    # Messages from the phone arrive with is_from_me=false.
    if message_dict.get("is_from_me", False):
        logger.debug("Skipping own message (is_from_me=true)")
        return False

    text = message_dict.get("text", "").strip()
    if not text:
        logger.debug("Skipping empty message")
        return False

    sender = message_dict.get("sender", IMESSAGE_TARGET)
    logger.info("Incoming message from %s: %s", sender, text[:80])

    reply = generate_reply(text, db_path=db_path, reports_dir=reports_dir)
    logger.info("Generated reply (%d chars)", len(reply))

    sent = False
    try:
        result = subprocess.run(
            ["imsg", "send", "--to", sender, "--text", reply],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Reply sent to %s", sender)
            sent = True
        else:
            logger.error("imsg send failed (exit %d): %s", result.returncode, result.stderr)
    except FileNotFoundError:
        logger.error("imsg CLI not found")
    except subprocess.TimeoutExpired:
        logger.error("imsg send timed out")
    except Exception as e:
        logger.error("Error sending reply: %s", e)

    # Process message for memory extraction (after reply, so user gets response fast)
    try:
        from research.user_memory import process_message
        process_message(db_path, text, sender)
    except Exception as e:
        logger.warning("Memory extraction failed (non-fatal): %s", e)

    return sent


def tail_watch_file(watch_file, db_path="research/research.db", reports_dir="~/reports",
                    stop_event=None):
    """Tail the NDJSON output file from a separate imsg watch launchd agent.

    The imsg watch process runs as its own launchd agent (with FDA) and writes
    NDJSON to a log file. This function tails that file and processes messages.

    Args:
        watch_file: Path to the NDJSON file written by imsg watch.
        db_path: Path to SQLite database.
        reports_dir: Path to reports directory.
        stop_event: threading.Event for clean shutdown.
    """
    logger.info("Tailing imsg watch output from %s", watch_file)

    # Wait for the file to appear
    while not os.path.exists(watch_file):
        if stop_event and stop_event.is_set():
            return
        logger.info("Waiting for watch file to appear: %s", watch_file)
        time.sleep(2)

    with open(watch_file, "r") as f:
        # Seek to end — only process new messages
        f.seek(0, 2)

        while stop_event is None or not stop_event.is_set():
            line = f.readline()
            if line:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    handle_message(msg, db_path=db_path, reports_dir=reports_dir)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON line: %s", line[:100])
                except Exception as e:
                    logger.error("Error handling message: %s", e)
            else:
                time.sleep(0.5)
