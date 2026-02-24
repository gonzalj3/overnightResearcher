"""Nightly research pipeline orchestrator."""

import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import date, datetime, timezone

import subprocess

import requests
import yaml

from research.db import (
    init_db,
    insert_daily_report,
    insert_source,
    update_interest_weights,
)
from research.fetcher import run_fetch_pipeline
from research.memory import build_memory_context
from research.report import format_report, save_report
from research.summarizer import run_batch_summarize
from research.synthesis import run_synthesis

logger = logging.getLogger("research.orchestrate")

OLLAMA_URL = "http://localhost:11434"
IMESSAGE_TARGET = "+12104265298"
IMESSAGE_CHANNEL = "bluebubbles"


def send_imessage_notification(report_path, executive_summary=""):
    """Send a brief iMessage with the report summary via OpenClaw + BlueBubbles.

    Args:
        report_path: Path to the saved report file.
        executive_summary: Short summary text to include in the message.

    Returns:
        True if the message was sent (or appeared to send), False otherwise.
    """
    date_str = date.today().isoformat()

    # Build a concise message (iMessage-friendly, not full report)
    lines = [f"Research Report Ready — {date_str}"]
    if executive_summary:
        # Coerce to string — LLM may return a list of bullet points
        if isinstance(executive_summary, list):
            executive_summary = "\n".join(
                f"- {item}" if isinstance(item, str) else f"- {str(item)}"
                for item in executive_summary
            )
        elif not isinstance(executive_summary, str):
            executive_summary = str(executive_summary)
        # Truncate to ~500 chars to keep the message brief
        summary = executive_summary[:500]
        if len(executive_summary) > 500:
            summary = summary.rsplit(" ", 1)[0] + "..."
        lines.append("")
        lines.append(summary)
    lines.append("")
    lines.append(f"Full report: {report_path}")

    message = "\n".join(lines)

    try:
        result = subprocess.run(
            [
                "openclaw", "message", "send",
                "--channel", IMESSAGE_CHANNEL,
                "-t", IMESSAGE_TARGET,
                "-m", message,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # BlueBubbles may return exit code 1 with a 500 error
        # but still deliver the message successfully
        if result.returncode == 0:
            logger.info("iMessage notification sent successfully")
            return True
        elif "BlueBubbles send failed (500)" in result.stderr:
            logger.warning(
                "iMessage returned 500 but may have delivered (known BlueBubbles quirk)"
            )
            return True
        else:
            logger.error("iMessage send failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error("openclaw CLI not found — cannot send iMessage")
        return False
    except subprocess.TimeoutExpired:
        logger.error("iMessage send timed out after 30s")
        return False
    except Exception as e:
        logger.error("iMessage send error: %s", e)
        return False


def setup_logging(log_dir=None):
    """Configure logging to file and stderr."""
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    date_str = date.today().isoformat()
    log_path = os.path.join(log_dir, f"{date_str}.log")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger("research")
    root_logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers on re-entry
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return log_dir


def run_health_check():
    """Check that all required services are available.

    Returns dict with status of each check.
    """
    result = {
        "ollama_running": False,
        "ram_available_gb": 0.0,
        "disk_available_gb": 0.0,
        "model_responding": False,
    }

    # Check Ollama
    for attempt in range(3):
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
            if resp.status_code == 200:
                result["ollama_running"] = True
                break
        except (requests.ConnectionError, requests.Timeout):
            if attempt < 2:
                time.sleep(10)

    # Check RAM (macOS)
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip()
        total_bytes = int(out)
        # Use vm_stat for free pages
        vm_out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 16384  # default on Apple Silicon
        free_pages = 0
        for line in vm_out.splitlines():
            if "Pages free" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        result["ram_available_gb"] = round((free_pages * page_size) / (1024**3), 1)
    except Exception:
        # Fallback: assume enough RAM
        result["ram_available_gb"] = 8.0

    # Check disk space
    try:
        usage = shutil.disk_usage(os.path.expanduser("~"))
        result["disk_available_gb"] = round(usage.free / (1024**3), 1)
    except Exception:
        result["disk_available_gb"] = 10.0

    # Check model responds
    if result["ollama_running"]:
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "qwen3:32b",
                    "prompt": "/no_think Say OK",
                    "stream": False,
                    "options": {"num_ctx": 512},
                },
                timeout=60,
            )
            if resp.status_code == 200 and resp.json().get("response"):
                result["model_responding"] = True
        except Exception:
            pass

    return result


def _load_manifest_for_summarizer(raw_dir):
    """Convert fetcher output dir into the manifest format summarizer expects.

    The fetcher writes a flat list manifest. The summarizer expects:
    {"date": "...", "items": [{"url", "content_hash", "content_path"}]}
    """
    manifest_path = os.path.join(raw_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return {"date": date.today().isoformat(), "items": []}

    with open(manifest_path) as f:
        raw_manifest = json.load(f)

    date_str = os.path.basename(raw_dir)
    items = []
    for i, entry in enumerate(raw_manifest):
        # Determine content file path — match the naming pattern from fetcher.py
        content_hash = entry.get("content_hash", "")
        source_type = entry.get("source_type", "unknown")
        # Files are named {type}_{index}.md but we don't know the exact index.
        # Scan dir for files matching the hash or just use index ordering.
        # Simpler: reconstruct from the directory listing.
        content_path = ""
        for fname in os.listdir(raw_dir):
            if fname == "manifest.json":
                continue
            fpath = os.path.join(raw_dir, fname)
            if os.path.isfile(fpath):
                # Check content hash
                import hashlib
                with open(fpath) as cf:
                    file_content = cf.read()
                fhash = hashlib.sha256(file_content.encode()).hexdigest()
                if fhash == content_hash:
                    content_path = fpath
                    break

        if content_path:
            items.append({
                "url": entry.get("url", ""),
                "content_hash": content_hash,
                "content_path": content_path,
                "source_type": source_type,
                "title": entry.get("title", ""),
            })

    return {"date": date_str, "items": items}


def _load_summaries_from_dir(summaries_dir):
    """Load all summary JSON files from a directory."""
    summaries = []
    if not os.path.isdir(summaries_dir):
        return summaries
    for fname in os.listdir(summaries_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(summaries_dir, fname)
        try:
            with open(fpath) as f:
                summary = json.load(f)
            summaries.append(summary)
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def run_nightly_research(
    sources_override=None,
    db_path=None,
    output_dir=None,
    log_dir=None,
    sources_path=None,
):
    """Run the full nightly research pipeline.

    Args:
        sources_override: dict with optional "max_total" and "include_broken_urls"
        db_path: path to SQLite database
        output_dir: path for report output
        log_dir: path for log files
        sources_path: path to sources.yaml

    Returns path to the generated report.
    """
    start_time = time.time()

    # Setup
    log_dir = setup_logging(log_dir)
    logger.info("=== Nightly Research Pipeline Starting ===")

    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "research.db")
    if output_dir is None:
        output_dir = os.path.expanduser("~/reports")
    if sources_path is None:
        sources_path = os.path.join(os.path.dirname(__file__), "sources.yaml")

    sources_override = sources_override or {}
    max_total = sources_override.get("max_total")

    try:
        # Step 1: Initialize DB
        logger.info("Step 1: Initializing database at %s", db_path)
        init_db(db_path)

        # Load config for interests
        with open(sources_path) as f:
            config = yaml.safe_load(f)
        interests_config = config.get("interests", {})
        interests = interests_config.get("primary", []) + interests_config.get("secondary", [])
        if not interests:
            interests = ["AI agents", "local LLM inference"]

        # Step 2: Fetch
        logger.info("Step 2: Fetching sources (max_total=%s)", max_total)
        raw_dir = run_fetch_pipeline(
            sources_path=sources_path,
            max_total=max_total,
        )
        # Count fetched items
        manifest_path = os.path.join(raw_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                raw_manifest = json.load(f)
            total_fetched = len(raw_manifest)
        else:
            total_fetched = 0
        logger.info("Fetched %d sources to %s", total_fetched, raw_dir)

        # Step 3: Summarize
        logger.info("Step 3: Summarizing sources")
        manifest = _load_manifest_for_summarizer(raw_dir)
        summary_results = run_batch_summarize(
            manifest, interests=interests
        )
        logger.info(
            "Summarized: %d succeeded, %d failed, %d skipped",
            summary_results["total_succeeded"],
            summary_results["total_failed"],
            summary_results["total_skipped"],
        )

        # Step 4: Build memory context
        logger.info("Step 4: Building memory context")
        memory = build_memory_context(db_path)

        # Step 5: Synthesize report
        logger.info("Step 5: Synthesizing report")
        summaries = _load_summaries_from_dir(summary_results["summaries_dir"])
        if not summaries:
            logger.warning("No summaries to synthesize — producing empty report")
            summaries = [{
                "title": "No data",
                "summary": "No sources were successfully summarized.",
                "relevance_score": 0.0,
                "relevance_tags": [],
                "key_developments": [],
            }]

        synthesis_result = run_synthesis(summaries, memory, output_dir=output_dir)
        report_path = synthesis_result["report_path"]
        logger.info("Report saved to %s", report_path)

        # Step 6: Persist to DB
        logger.info("Step 6: Persisting results to database")
        # Insert each summary as a source record
        for summary in summaries:
            insert_source(db_path, {
                "url": summary.get("source_url", summary.get("url", "")),
                "content_hash": summary.get("content_hash", ""),
                "source_type": summary.get("source_type", "unknown"),
                "fetched_at": summary.get("summarized_at", datetime.now(timezone.utc).isoformat()),
                "title": summary.get("title", ""),
                "summary": summary.get("summary", ""),
                "relevance_score": summary.get("relevance_score", 0.0),
                "relevance_tags": json.dumps(summary.get("relevance_tags", [])) if isinstance(summary.get("relevance_tags"), list) else summary.get("relevance_tags", "[]"),
                "key_developments": json.dumps(summary.get("key_developments", [])) if isinstance(summary.get("key_developments"), list) else summary.get("key_developments", "[]"),
                "raw_content_path": summary.get("content_path", ""),
            })

        # Update interest weights from tags
        tag_counts = {}
        for summary in summaries:
            if summary.get("relevance_score", 0) >= 0.5:
                tags = summary.get("relevance_tags", [])
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        for tag, count in tag_counts.items():
            update_interest_weights(db_path, tag, hits=count)

        # Insert daily report record
        insert_daily_report(db_path, {
            "report_date": date.today().isoformat(),
            "report_path": report_path,
            "total_sources": len(summaries),
            "top_developments": json.dumps([
                s.get("title", "") for s in sorted(
                    summaries,
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True,
                )[:5]
            ]),
            "interests_snapshot": json.dumps(interests),
        })

        # Step 7: Send iMessage notification
        logger.info("Step 7: Sending iMessage notification")
        executive_summary = synthesis_result.get("executive_summary", "")
        send_imessage_notification(report_path, executive_summary)

        elapsed = time.time() - start_time
        logger.info(
            "=== Pipeline complete in %.1f minutes. Report: %s ===",
            elapsed / 60,
            report_path,
        )
        return report_path

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "=== Pipeline FAILED after %.1f minutes ===", elapsed / 60
        )
        logger.error("Error: %s", e)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Entry point for cron / launchd
    report = run_nightly_research()
    print(f"Report: {report}")
