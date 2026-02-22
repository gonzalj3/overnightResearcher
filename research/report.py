"""Report formatter and file writer."""

import os
from datetime import date


def format_report(report_data):
    """Produce a markdown report from structured report data."""
    d = report_data.get("date", date.today().isoformat())
    sections = [f"# AI Research Report — {d}"]

    # Executive Summary
    sections.append("## Executive Summary")
    sections.append(report_data.get("executive_summary", "No summary available."))

    # Watch List
    watch = report_data.get("watch_list", [])
    if watch:
        sections.append("## Watch List")
        for item in watch:
            sections.append(f"- {item}")

    # Detailed Findings
    themes = report_data.get("themes", [])
    if themes:
        sections.append("## Detailed Findings")
        for theme in themes:
            sections.append(f"### {theme.get('theme', 'Topic')}")
            sections.append(theme.get("analysis", ""))
            for src in theme.get("sources", []):
                if src:
                    sections.append(f"- Source: {src}")

    # Statistics
    stats = report_data.get("stats", {})
    if stats:
        sections.append("## Statistics")
        sections.append(f"- Sources processed: {stats.get('sources_processed', 0)}")
        sections.append(f"- Sources skipped (dedup): {stats.get('sources_skipped', 0)}")
        sections.append(f"- Fetch errors: {stats.get('fetch_errors', 0)}")
        sections.append(f"- Processing time: {stats.get('processing_time', 'N/A')}")
        drift = stats.get("interest_drift")
        if drift:
            sections.append(f"- Interest drift: {drift}")

    # Sources list
    all_sources = []
    for theme in themes:
        all_sources.extend(s for s in theme.get("sources", []) if s)
    if all_sources:
        sections.append("## Sources")
        for src in sorted(set(all_sources)):
            sections.append(f"- {src}")

    return "\n\n".join(sections) + "\n"


def save_report(report_md, output_dir=None, report_date=None):
    """Save markdown report to disk. Returns the file path."""
    if output_dir is None:
        output_dir = os.path.expanduser("~/reports")
    if report_date is None:
        report_date = date.today().isoformat()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{report_date}.md")
    with open(path, "w") as f:
        f.write(report_md)
    return path
