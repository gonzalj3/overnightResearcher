"""Report synthesis pipeline: 3-pass LLM-driven analysis."""

import json
import logging
import os
from datetime import date

import requests

from research.json_repair import repair_json

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:32b"


def _llm_call(prompt, system="You are a research analyst. Output valid JSON only.",
               use_json=True, timeout=180):
    """Make a single Ollama call."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    if use_json:
        payload["format"] = "json"

    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json().get("response", "")

    if use_json:
        return repair_json(raw)
    return raw


def cluster_and_rank(summaries, top_n=10):
    """Pass 1: Cluster summaries by theme and rank top items.

    Returns list of clusters: [{"theme": str, "items": [summary_dicts]}]
    """
    # Build compact input
    compact = []
    for s in sorted(summaries, key=lambda x: x.get("relevance_score", 0), reverse=True):
        compact.append({
            "title": s.get("title", "Untitled")[:80],
            "summary": s.get("summary", "")[:100],
            "score": s.get("relevance_score", 0),
            "tags": s.get("relevance_tags", [])[:3],
        })
    compact = compact[:top_n * 2]  # send more than top_n for grouping

    prompt = f"""/no_think
Group these research items by theme. Select the top {top_n} most relevant items.
Return JSON: {{"clusters": [{{"theme": "Theme Name", "items": [{{"title": "...", "summary": "..."}}]}}]}}

Items:
{json.dumps(compact, indent=1)}"""

    result = _llm_call(prompt)

    if result.get("parse_failed"):
        # Fallback: single cluster with top items
        return [{"theme": "General Findings", "items": summaries[:top_n]}]

    clusters = result.get("clusters", [])
    if not clusters:
        return [{"theme": "General Findings", "items": summaries[:top_n]}]

    # Enrich cluster items with full summary data
    title_map = {s.get("title", ""): s for s in summaries}
    for cluster in clusters:
        enriched = []
        for item in cluster.get("items", []):
            title = item.get("title", "")
            full = title_map.get(title)
            if full:
                enriched.append(full)
            else:
                enriched.append(item)
        cluster["items"] = enriched

    return clusters


def synthesize_theme(cluster, memory_context=""):
    """Pass 2: Produce analysis paragraph for a single theme cluster.

    Returns analysis string.
    """
    items_text = "\n".join(
        f"- {item.get('title', 'Untitled')}: {item.get('summary', 'No summary')}"
        for item in cluster.get("items", [])
    )
    theme = cluster.get("theme", "Research Findings")

    prompt = f"""/no_think
Write a concise analysis paragraph (3-5 sentences) about this research theme.

Theme: {theme}

Sources:
{items_text}

{"Historical context: " + memory_context if memory_context else ""}

Respond with JSON: {{"analysis": "Your analysis paragraph here."}}"""

    result = _llm_call(prompt)
    if isinstance(result, dict):
        return result.get("analysis", result.get("summary", str(result)))
    return str(result)


def generate_executive_summary(theme_analyses, memory_context=""):
    """Pass 3: Produce executive summary and watch list.

    Returns dict with "executive_summary" and "watch_list".
    """
    themes_text = "\n\n".join(
        f"### {ta.get('theme', 'Topic')}\n{ta.get('analysis', '')}"
        for ta in theme_analyses
    )

    prompt = f"""/no_think
Based on these research theme analyses, produce:
1. An executive summary (3-5 bullet points of the most important findings)
2. A watch list (2-4 items/trends to follow up on tomorrow)

{themes_text}

{"Historical context: " + memory_context if memory_context else ""}

Respond with JSON: {{"executive_summary": "bullet points here", "watch_list": ["item1", "item2"]}}"""

    result = _llm_call(prompt)
    if result.get("parse_failed"):
        return {
            "executive_summary": "Report synthesis failed — see detailed findings below.",
            "watch_list": [],
        }

    return {
        "executive_summary": result.get("executive_summary", ""),
        "watch_list": result.get("watch_list", []),
    }


def run_synthesis(summaries, memory_context="", output_dir=None):
    """Full 3-pass synthesis pipeline.

    Returns dict with report_path, themes_count, executive_summary.
    """
    from research.report import format_report, save_report

    if output_dir is None:
        output_dir = os.path.expanduser("~/reports")

    # Pass 1: Cluster & rank
    logger.info("Pass 1: Clustering %d summaries", len(summaries))
    clusters = cluster_and_rank(summaries)

    # Pass 2: Synthesize each theme (max 5)
    logger.info("Pass 2: Synthesizing %d themes", len(clusters))
    theme_analyses = []
    for cluster in clusters[:5]:
        analysis = synthesize_theme(cluster, memory_context)
        sources = [
            item.get("url", item.get("source_url", ""))
            for item in cluster.get("items", [])
            if item.get("url") or item.get("source_url")
        ]
        theme_analyses.append({
            "theme": cluster.get("theme", "Topic"),
            "analysis": analysis,
            "sources": sources,
        })

    # Pass 3: Executive summary
    logger.info("Pass 3: Generating executive summary")
    exec_result = generate_executive_summary(theme_analyses, memory_context)

    # Format and save
    report_data = {
        "date": date.today().isoformat(),
        "executive_summary": exec_result["executive_summary"],
        "watch_list": exec_result.get("watch_list", []),
        "themes": theme_analyses,
        "stats": {
            "sources_processed": len(summaries),
            "sources_skipped": 0,
            "fetch_errors": 0,
            "processing_time": "00:00:00",
        },
    }

    report_md = format_report(report_data)
    report_path = save_report(report_md, output_dir=output_dir)

    return {
        "report_path": report_path,
        "themes_count": len(theme_analyses),
        "executive_summary": exec_result["executive_summary"],
    }
