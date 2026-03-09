"""Report synthesis pipeline: 3-pass LLM-driven analysis."""

import json
import logging
import os
from datetime import date

from research.gpu_timesheet import ollama_generate
from research.json_repair import repair_json

logger = logging.getLogger(__name__)

MODEL = "qwen3.5:27b"


def _llm_call(prompt, system="You are a research analyst. Output valid JSON only.",
               use_json=True, timeout=600, caller="synthesis"):
    """Make a single Ollama call."""
    raw = ollama_generate(
        model=MODEL,
        prompt=prompt,
        caller=caller,
        system=system,
        use_json=use_json,
        timeout=timeout,
        options={"temperature": 0.2, "num_ctx": 16384},
    )

    if use_json:
        return repair_json(raw)
    return raw


def cluster_and_rank(summaries, top_n=10, stale_hashes=None, focus_tags=None):
    """Pass 1: Cluster summaries by theme and rank top items.

    Args:
        summaries: list of summary dicts
        top_n: max items to select
        stale_hashes: set of content_hashes reported 3+ days ago (score * 0.3 penalty)
        focus_tags: list of tags to boost (score * 1.5 for matching items)

    Returns list of clusters: [{"theme": str, "items": [summary_dicts]}]
    """
    stale_hashes = stale_hashes or set()
    focus_tags_lower = {t.lower() for t in (focus_tags or [])}

    # Apply staleness penalty and focus boost to scores before ranking
    scored = []
    for s in summaries:
        score = s.get("relevance_score", 0)
        content_hash = s.get("content_hash", "")
        if content_hash in stale_hashes:
            score *= 0.3
        if focus_tags_lower:
            item_tags = s.get("relevance_tags", [])
            if isinstance(item_tags, str):
                try:
                    item_tags = json.loads(item_tags)
                except (json.JSONDecodeError, TypeError):
                    item_tags = []
            if any(t.lower() in focus_tags_lower for t in item_tags):
                score *= 1.5
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build compact input
    compact = []
    for score, s in scored:
        compact.append({
            "title": s.get("title", "Untitled")[:80],
            "summary": s.get("summary", "")[:100],
            "score": round(score, 3),
            "tags": s.get("relevance_tags", [])[:3],
        })
    compact = compact[:top_n * 2]  # send more than top_n for grouping

    prompt = f"""Group these research items by theme. Select the top {top_n} most relevant items.
Return JSON: {{"clusters": [{{"theme": "Theme Name", "items": [{{"title": "...", "summary": "..."}}]}}]}}

Items:
{json.dumps(compact, indent=1)}"""

    result = _llm_call(prompt, caller="synthesis.cluster")

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

    prompt = f"""Write a concise analysis paragraph (3-5 sentences) about this research theme.

Theme: {theme}

Sources:
{items_text}

{"Historical context: " + memory_context if memory_context else ""}

Respond with JSON: {{"analysis": "Your analysis paragraph here."}}"""

    result = _llm_call(prompt, caller="synthesis.theme")
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

    prompt = f"""Based on these research theme analyses, produce:
1. An executive summary (3-5 bullet points of the most important findings)
2. A watch list (2-4 items/trends to follow up on tomorrow)

{themes_text}

{"Historical context: " + memory_context if memory_context else ""}

Respond with JSON: {{"executive_summary": "bullet points here", "watch_list": ["item1", "item2"]}}"""

    result = _llm_call(prompt, caller="synthesis.executive")
    if result.get("parse_failed"):
        return {
            "executive_summary": "Report synthesis failed — see detailed findings below.",
            "watch_list": [],
        }

    return {
        "executive_summary": result.get("executive_summary", ""),
        "watch_list": result.get("watch_list", []),
    }


def run_synthesis(summaries, memory_context="", output_dir=None,
                  stale_hashes=None, focus_tags=None):
    """Full 3-pass synthesis pipeline.

    Returns dict with report_path, themes_count, executive_summary.
    """
    from research.report import format_report, save_report

    if output_dir is None:
        output_dir = os.path.expanduser("~/reports")

    # Pass 1: Cluster & rank
    logger.info("Pass 1: Clustering %d summaries", len(summaries))
    clusters = cluster_and_rank(summaries, stale_hashes=stale_hashes, focus_tags=focus_tags)

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
