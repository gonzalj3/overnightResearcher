"""LLM summarization pipeline using Ollama directly."""

import json
import logging
import os
import time
from datetime import date, datetime, timezone

from research.gpu_timesheet import ollama_generate
from research.json_repair import repair_json

logger = logging.getLogger(__name__)

MODEL = "qwen3:32b"
DEFAULT_INTERESTS = ["AI agents", "local LLM inference"]

SYSTEM_PROMPT = "You are a research summarizer. Output valid JSON only."

USER_TEMPLATE = """/no_think
Summarize this content. Respond with ONLY this JSON:
{{"title":"...", "summary":"2-3 sentences", "relevance_tags":["tag1","tag2"], "relevance_score": 0.0-1.0, "key_developments":["point1","point2"]}}

User interests: {interests}

Content:
{content}"""


def build_prompt(content, interests):
    """Build the summarization prompt."""
    interests_str = ", ".join(interests) if interests else "general AI research"
    return USER_TEMPLATE.format(interests=interests_str, content=content)


def summarize_source(content, interests, source_url="", timeout=120):
    """Summarize a single source via Ollama.

    Returns dict with title, summary, relevance_tags, relevance_score, key_developments.
    """
    if not content or not content.strip():
        return {
            "title": "Empty content",
            "summary": "No content to summarize.",
            "relevance_tags": [],
            "relevance_score": 0.0,
            "key_developments": [],
            "source_url": source_url,
        }

    prompt = build_prompt(content, interests)

    try:
        raw_response = ollama_generate(
            model=MODEL,
            prompt=prompt,
            caller="summarizer",
            system=SYSTEM_PROMPT,
            use_json=True,
            timeout=timeout,
            options={"temperature": 0.1, "num_ctx": 8192},
        )
    except Exception as e:
        logger.error("Ollama request failed for %s: %s", source_url, e)
        return {
            "title": "Request failed",
            "summary": f"Ollama request failed: {e}",
            "relevance_tags": [],
            "relevance_score": 0.0,
            "key_developments": [],
            "source_url": source_url,
            "error": str(e),
        }

    result = repair_json(raw_response)
    result["source_url"] = source_url

    # Ensure required fields exist with defaults
    result.setdefault("title", "Untitled")
    result.setdefault("summary", "")
    result.setdefault("relevance_tags", [])
    result.setdefault("relevance_score", 0.0)
    result.setdefault("key_developments", [])

    # Clamp relevance_score
    try:
        result["relevance_score"] = max(0.0, min(1.0, float(result["relevance_score"])))
    except (ValueError, TypeError):
        result["relevance_score"] = 0.0

    return result


def run_batch_summarize(manifest, interests=None, summaries_base_dir=None,
                        delay_between=0):
    """Batch summarize all items in a manifest.

    Args:
        manifest: dict with "date" and "items" list. Each item has
                  "url", "content_hash", and "content_path".
        interests: list of interest strings
        summaries_base_dir: base dir for summaries (default: research/summaries)
        delay_between: seconds to wait between calls (thermal throttling)

    Returns dict with total_processed, total_succeeded, total_failed,
    total_skipped, summaries_dir.
    """
    if interests is None:
        interests = DEFAULT_INTERESTS
    if summaries_base_dir is None:
        summaries_base_dir = "research/summaries"

    date_str = manifest.get("date", date.today().isoformat())
    summaries_dir = os.path.join(summaries_base_dir, date_str)
    os.makedirs(summaries_dir, exist_ok=True)

    items = manifest.get("items", [])
    total_processed = 0
    total_succeeded = 0
    total_failed = 0
    total_skipped = 0

    for i, item in enumerate(items):
        content_hash = item.get("content_hash", "unknown")
        summary_path = os.path.join(summaries_dir, f"{content_hash}.json")

        # Resume: skip already-processed
        if os.path.isfile(summary_path):
            total_skipped += 1
            logger.info("Skipping %d/%d (already summarized): %s",
                        i + 1, len(items), item.get("url", ""))
            continue

        # Read content
        content_path = item.get("content_path", "")
        if not content_path or not os.path.isfile(content_path):
            logger.warning("Content file missing: %s", content_path)
            total_failed += 1
            total_processed += 1
            continue

        with open(content_path) as f:
            content = f.read()

        logger.info("Summarizing %d/%d: %s", i + 1, len(items), item.get("url", ""))
        result = summarize_source(content, interests, source_url=item.get("url", ""))
        result["content_hash"] = content_hash
        result["summarized_at"] = datetime.now(timezone.utc).isoformat()

        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)

        if result.get("parse_failed") or result.get("error"):
            total_failed += 1
        else:
            total_succeeded += 1
        total_processed += 1

        if delay_between > 0 and i < len(items) - 1:
            time.sleep(delay_between)

    return {
        "total_processed": total_processed,
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "summaries_dir": summaries_dir,
    }
