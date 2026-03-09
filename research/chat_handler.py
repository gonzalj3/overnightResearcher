"""Two-way iMessage chat handler via imsg watch."""

import json
import logging
import os
import re
import subprocess
import time
from datetime import date, datetime, timedelta

import requests

from research.db import get_recent_sources
from research.gpu_timesheet import ollama_generate
from research.memory import build_memory_context
from research.tools import run_tool_loop, TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

MODEL = "qwen3.5:9b"
IMESSAGE_TARGET = "+12104265298"
PHONE_CHAT_ID = 4   # chat with phone number identifier — agent sends replies here
EMAIL_CHAT_ID = 5    # chat with email identifier — user sends questions from phone here
MAX_REPLY_CHARS = 1500

# Track recently sent replies to detect echoes on chat_id 5.
# When agent sends a reply, it appears on chat_id 4 AND echoes on chat_id 5.
# We store the first 100 chars of each sent reply to filter echoes.
_recent_replies = []  # list of (timestamp, text_prefix) tuples
_REPLY_CACHE_MAX = 20
_REPLY_CACHE_TTL = 120  # seconds


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


def _get_recent_fetches(db_path, hours=6, limit=15):
    """Pull the most recent high-relevance sources from the DB.

    Returns a formatted string of recent fetches, or empty string if none.
    """
    try:
        # get_recent_sources uses days, so we fetch 1 day and filter by hours
        recent = get_recent_sources(db_path, days=1)
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        fresh = [
            s for s in recent
            if s.get("fetched_at", "") >= cutoff
        ]
        # Sort by relevance score descending
        fresh.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        fresh = fresh[:limit]

        if not fresh:
            return ""

        lines = []
        for s in fresh:
            title = s.get("title", "Untitled")[:80]
            summary = s.get("summary", "")[:150]
            score = s.get("relevance_score", 0)
            lines.append(f"- [{score:.1f}] {title}: {summary}")

        return "\n".join(lines)
    except Exception as e:
        logger.debug("Could not load recent fetches: %s", e)
        return ""


_URL_RE = re.compile(r'https?://[^\s<>"\']+')
_FETCH_INTENT_RE = re.compile(
    r'\b(?:pull|fetch|grab|go\s+to|check|what\'?s\s+on|look\s+at|read|open|visit|scrape|get)\b',
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|org|net|io|dev|ai|co|me|info|site|blog|news|xyz))\b')
_HN_RE = re.compile(r'\b(?:hacker\s*news|HN)\b', re.IGNORECASE)


def detect_fetch_request(text):
    """Detect if the user wants us to fetch a URL or website.

    Returns dict with:
      - should_fetch: bool
      - url: explicit URL if found
      - site: domain/site name if found (no full URL)
      - source: "hn" for Hacker News requests
      - topic: optional topic filter (for HN)
      - query: the original user text
    """
    result = {"should_fetch": False, "query": text}

    # Check for explicit URL
    url_match = _URL_RE.search(text)
    if url_match:
        result["should_fetch"] = True
        result["url"] = url_match.group(0).rstrip(".,;:!?)")
        return result

    # Check for Hacker News
    hn_match = _HN_RE.search(text)
    if hn_match:
        result["should_fetch"] = True
        result["source"] = "hn"
        # Extract topic: everything after HN mention, minus filler words
        after = text[hn_match.end():].strip()
        topic = re.sub(r'^(?:about|for|on|regarding)\s+', '', after, flags=re.IGNORECASE).strip()
        topic = re.sub(r'[?.!]+$', '', topic).strip()
        result["topic"] = topic if topic else None
        return result

    # Check for fetch intent + domain name
    if _FETCH_INTENT_RE.search(text):
        domain_match = _DOMAIN_RE.search(text)
        if domain_match:
            result["should_fetch"] = True
            result["site"] = domain_match.group(1)
            return result

    return result


def handle_fetch_request(fetch_info, db_path="research/research.db"):
    """Handle a detected fetch request by fetching content and summarizing.

    Args:
        fetch_info: dict from detect_fetch_request()
        db_path: path to SQLite database

    Returns:
        Reply string for the user.
    """
    source = fetch_info.get("source")
    url = fetch_info.get("url")
    site = fetch_info.get("site")
    query = fetch_info.get("query", "")
    topic = fetch_info.get("topic")

    if source == "hn":
        return _handle_hn_request(topic, query)
    elif url:
        return _handle_url_request(url, query)
    elif site:
        return _handle_url_request(f"https://{site}", query)
    else:
        return "I couldn't figure out what to fetch. Try sending a URL or site name."


def _handle_hn_request(topic, query):
    """Fetch and format Hacker News stories, optionally filtered by topic."""
    from research.fetch_hn import fetch_hn_stories

    stories = fetch_hn_stories(30)
    if not stories:
        return "Couldn't reach Hacker News right now. Try again in a bit."

    if topic:
        # Use 8B model to filter stories by topic
        stories_text = "\n".join(
            f"- {s['title']} (score: {s['score']})" for s in stories
        )
        prompt = f"""From these Hacker News stories, pick the ones related to "{topic}". Return ONLY a JSON array of the matching story titles, e.g. ["title1", "title2"]. If none match, return [].

{stories_text}"""
        try:
            raw = ollama_generate(
                model=MODEL,
                prompt=prompt,
                caller="chat_handler.hn_filter",
                use_json=True,
                timeout=60,
                options={"temperature": 0.1, "num_ctx": 4096},
            ) or "[]"
            # Parse the JSON array of titles
            try:
                matched_titles = json.loads(raw)
                if isinstance(matched_titles, dict):
                    # LLM might wrap in {"titles": [...]}
                    matched_titles = list(matched_titles.values())[0] if matched_titles else []
                if isinstance(matched_titles, list) and matched_titles:
                    title_set = {t.lower() for t in matched_titles if isinstance(t, str)}
                    filtered = [s for s in stories if s["title"].lower() in title_set]
                    if filtered:
                        stories = filtered
                    # If no exact match, fall back to all stories
            except (json.JSONDecodeError, TypeError):
                pass  # Fall through to top-by-score
        except Exception as e:
            logger.warning("LLM filtering failed, showing top stories: %s", e)

    # Sort by score, take top 5
    stories.sort(key=lambda s: s.get("score", 0), reverse=True)
    top = stories[:5]
    header = f"Top HN stories about \"{topic}\"" if topic else "Top Hacker News stories right now"
    return _format_hn_stories(top, header)


def _format_hn_stories(stories, header):
    """Format HN stories as a numbered list for iMessage."""
    lines = [header, ""]
    for i, s in enumerate(stories, 1):
        title = s.get("title", "Untitled")
        score = s.get("score", 0)
        url = s.get("url", "")
        hn_link = s.get("comments_url", "")
        line = f"{i}. {title} ({score} pts)"
        if url:
            line += f"\n   {url}"
        if hn_link:
            line += f"\n   Discussion: {hn_link}"
        lines.append(line)
    return "\n".join(lines)


def _handle_url_request(url, query):
    """Fetch a URL and summarize the content."""
    from research.fetch_http import fetch_url

    content = fetch_url(url)
    if not content:
        return f"Couldn't fetch {url} — the site may be down or blocking requests."

    return _summarize_fetched_content(content, query, url)


def _summarize_fetched_content(content, query, url=""):
    """Send fetched content to 8B model for extraction/summarization."""
    # Truncate to fit in context
    truncated = content[:4000]
    if len(content) > 4000:
        truncated += "\n...(truncated)"

    prompt = f"""You fetched this web page{' from ' + url if url else ''}. The user asked: "{query}"

Extract the most relevant information and summarize it concisely for iMessage (2-3 short paragraphs). Include any key links, names, or numbers.

=== Page Content ===
{truncated}"""

    try:
        reply = ollama_generate(
            model=MODEL,
            prompt=prompt,
            caller="chat_handler.fetch_summarize",
            timeout=120,
            options={"temperature": 0.3, "num_ctx": 8192},
        ).strip()
        return _truncate_reply(reply)
    except Exception as e:
        logger.error("Failed to summarize fetched content: %s", e)
        # Return raw content excerpt as fallback
        fallback = content[:500]
        return f"Here's what I found at {url}:\n\n{fallback}..."


def generate_reply_with_tools(question, db_path="research/research.db",
                              reports_dir="~/reports"):
    """Generate a reply using LLM tool calling.

    The model decides autonomously which tools to call (fetch URLs, query DB, etc.)
    based on the user's question. Falls back to None if tool calling fails.

    Returns reply string, or None on failure.
    """
    system = (
        "You are a helpful AI research assistant. You have tools to fetch web pages, "
        "check Hacker News, read GitHub READMEs, and search a local research database. "
        "Use tools when the user asks for live data. For general questions, answer directly. "
        "Keep replies concise (2-3 short paragraphs), suitable for iMessage."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    try:
        reply = run_tool_loop(
            model=MODEL,
            messages=messages,
            caller="chat_handler.tools",
            db_path=db_path,
        )
        if reply:
            return _truncate_reply(reply)
        return None
    except Exception as e:
        logger.warning("Tool-calling path failed: %s", e)
        return None


def generate_reply(question, db_path="research/research.db", reports_dir="~/reports"):
    """Generate a reply to a user question using Ollama 8B.

    Reads today's report, recent DB fetches, and memory context, then asks the LLM.

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

    # Pull recent fetches from DB (includes 3-hour cycle results)
    recent_fetches = _get_recent_fetches(db_path)

    # Truncate report to leave room for prompt + response
    report_excerpt = report_text[:3000] if report_text else "No report available today."

    prompt = f"""You are a helpful AI research assistant. Answer the user's question based on the research data below. Keep your answer concise (2-3 short paragraphs) and iMessage-friendly.

{user_profile}

=== Latest Fetches (last 6 hours) ===
{recent_fetches if recent_fetches else "No recent fetches."}

=== Today's Research Report ===
{report_excerpt}

=== Historical Context ===
{memory_context if memory_context else "No historical context available."}

=== User Question ===
{question}"""

    try:
        reply = ollama_generate(
            model=MODEL,
            prompt=prompt,
            caller="chat_handler.reply",
            timeout=120,
            options={"temperature": 0.3, "num_ctx": 8192},
        ).strip()
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
    # Filter out agent replies (messages sent by this Mac).
    # chat_id 4 (phone thread): is_from_me reliably distinguishes direction.
    # chat_id 5 (email thread): both directions show is_from_me=true because
    # it's the same iCloud account. We detect echoed agent replies by matching
    # against recently sent reply text.
    chat_id = message_dict.get("chat_id")
    if chat_id != EMAIL_CHAT_ID and message_dict.get("is_from_me", False):
        logger.debug("Skipping own message (is_from_me=true, chat_id=%s)", chat_id)
        return False

    text = message_dict.get("text", "").strip()
    if not text:
        logger.debug("Skipping empty message")
        return False

    # Detect echoed agent replies on chat_id 5 by matching against recent sends.
    if chat_id == EMAIL_CHAT_ID:
        now = time.time()
        # Clean expired entries
        _recent_replies[:] = [
            (ts, prefix) for ts, prefix in _recent_replies
            if now - ts < _REPLY_CACHE_TTL
        ]
        # Strip any binary prefixes (imsg adds control chars) for comparison
        clean_text = text.lstrip('\x00\x01\x02\x03\x04\x05').strip()
        for _, prefix in _recent_replies:
            if clean_text[:80] == prefix[:80]:
                logger.debug("Skipping echoed agent reply on chat_id 5")
                return False

    sender = message_dict.get("sender", IMESSAGE_TARGET)
    logger.info("Incoming message from %s: %s", sender, text[:80])

    # Try tool-calling path first — model decides what to do
    reply = generate_reply_with_tools(text, db_path=db_path, reports_dir=reports_dir)

    # Fallback to manual routing if tool calling failed
    if not reply:
        logger.info("Tool-calling returned nothing, falling back to manual routing")
        fetch_req = detect_fetch_request(text)
        if fetch_req["should_fetch"]:
            reply = handle_fetch_request(fetch_req, db_path=db_path)
        else:
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
            # Cache reply text to detect echoes on chat_id 5
            _recent_replies.append((time.time(), reply.strip()[:100]))
            if len(_recent_replies) > _REPLY_CACHE_MAX:
                _recent_replies.pop(0)
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
