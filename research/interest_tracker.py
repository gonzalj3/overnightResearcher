"""Interest profile evolution and multi-day continuity."""

import json
import os
from datetime import date, timedelta

import yaml

from research.db import (
    get_interest_weights,
    get_recent_sources,
    get_trending_topics,
    update_interest_weights,
)


def adjust_weights(db_path, decay_factor=0.05):
    """Adjust interest weights based on recent relevance hits.

    - Interests with many high-relevance hits get boosted
    - Interests with no recent hits get slightly decayed
    - No interest is deleted, just lowered in priority

    Returns dict of {interest: new_weight}.
    """
    weights = get_interest_weights(db_path)
    weight_map = {w["interest"]: w for w in weights}

    # Count high-relevance sources per tag in last 7 days
    recent = get_recent_sources(db_path, days=7)
    tag_hits = {}
    for source in recent:
        if source.get("relevance_score", 0) < 0.7:
            continue
        tags_raw = source.get("relevance_tags", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except (json.JSONDecodeError, TypeError):
            tags = []
        for tag in tags:
            tag_lower = tag.lower()
            tag_hits[tag_lower] = tag_hits.get(tag_lower, 0) + 1

    # Boost interests that had high-relevance hits
    for interest, data in weight_map.items():
        interest_lower = interest.lower()
        if interest_lower in tag_hits:
            hits = tag_hits[interest_lower]
            update_interest_weights(db_path, interest, hits=hits)
        else:
            # Decay is now handled at read-time via get_effective_weights().
            # Stored weights only increase; effective weights decay over time
            # based on hours since last_seen_at (half-life ~6 days).
            pass

    return {w["interest"]: w["weight"] for w in get_interest_weights(db_path)}


def suggest_new_interests(db_path, current_interests, days=14, min_occurrences=3):
    """Suggest new interests based on recurring tags not in the current profile.

    Returns list of dicts: [{"tag": str, "count": int, "avg_score": float}]
    """
    current_lower = {i.lower() for i in current_interests}

    # Get all sources from the time window
    recent = get_recent_sources(db_path, days=days)

    # Aggregate tags
    tag_data = {}  # tag -> {"count": int, "total_score": float}
    for source in recent:
        tags_raw = source.get("relevance_tags", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except (json.JSONDecodeError, TypeError):
            tags = []
        score = source.get("relevance_score", 0.0)
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in current_lower:
                continue
            if tag_lower not in tag_data:
                tag_data[tag_lower] = {"count": 0, "total_score": 0.0, "tag": tag}
            tag_data[tag_lower]["count"] += 1
            tag_data[tag_lower]["total_score"] += score

    # Filter by minimum occurrences and sort by count
    suggestions = []
    for tag_lower, data in tag_data.items():
        if data["count"] >= min_occurrences:
            suggestions.append({
                "tag": data["tag"],
                "count": data["count"],
                "avg_score": round(data["total_score"] / data["count"], 2),
            })

    return sorted(suggestions, key=lambda x: x["count"], reverse=True)


def apply_overrides(weights, overrides):
    """Apply user overrides to interest weights.

    Args:
        weights: dict of {interest: weight}
        overrides: dict with keys "pinned", "blocked", "focus"
            - pinned: list of interests that should never have weight reduced
            - blocked: list of interests to remove entirely
            - focus: list of interests to temporarily boost (weight += 2.0)

    Returns new weights dict with overrides applied.
    """
    result = dict(weights)

    # Remove blocked interests
    for blocked in overrides.get("blocked", []):
        result.pop(blocked, None)

    # Ensure pinned interests are never reduced (keep at least current value)
    for pinned in overrides.get("pinned", []):
        if pinned in result:
            # Already at current value, nothing to reduce
            pass

    # Boost focus interests
    for focus in overrides.get("focus", []):
        if focus in result:
            result[focus] = result[focus] + 2.0
        else:
            result[focus] = 3.0  # new focus interest starts high

    return result


def load_overrides(overrides_path=None):
    """Load interest overrides from YAML file.

    Expected format:
    ```yaml
    pinned:
      - AI agents
    blocked:
      - cryptocurrency
    focus:
      - fusion energy
    ```
    """
    if overrides_path is None:
        overrides_path = os.path.join(
            os.path.dirname(__file__), "interest_overrides.yaml"
        )
    if not os.path.isfile(overrides_path):
        return {"pinned": [], "blocked": [], "focus": []}
    with open(overrides_path) as f:
        data = yaml.safe_load(f) or {}
    return {
        "pinned": data.get("pinned", []),
        "blocked": data.get("blocked", []),
        "focus": data.get("focus", []),
    }


def generate_weekly_digest(db_path, weeks=1):
    """Generate a weekly rollup digest from the last N weeks of data.

    Returns a markdown string summarizing the week.
    """
    days = weeks * 7
    recent = get_recent_sources(db_path, days=days)
    trending = get_trending_topics(db_path, days=days)
    weights = get_interest_weights(db_path)

    # Aggregate stats
    total_sources = len(recent)
    high_relevance = [s for s in recent if s.get("relevance_score", 0) >= 0.7]

    # Top developments (by score)
    top = sorted(recent, key=lambda x: x.get("relevance_score", 0), reverse=True)[:10]

    # Build digest
    lines = [
        f"# Weekly Research Digest",
        f"**Period:** {(date.today() - timedelta(days=days)).isoformat()} to {date.today().isoformat()}",
        "",
        "## Summary Statistics",
        f"- Total sources processed: {total_sources}",
        f"- High-relevance sources (>0.7): {len(high_relevance)}",
        "",
    ]

    if top:
        lines.append("## Top Developments")
        for item in top:
            title = item.get("title", "Untitled")[:80]
            score = item.get("relevance_score", 0)
            lines.append(f"- **{title}** (score: {score:.2f})")
        lines.append("")

    if trending:
        lines.append("## Trending Topics")
        for t in trending[:10]:
            lines.append(f"- {t['tag']}: {t['count']} mentions")
        lines.append("")

    if weights:
        lines.append("## Interest Weight Changes")
        for w in weights[:10]:
            lines.append(
                f"- {w['interest']}: weight={w['weight']:.1f} "
                f"(total hits: {w['total_hits']})"
            )
        lines.append("")

    return "\n".join(lines)


def generate_drift_report(db_path):
    """Show how interests changed over the last run.

    Returns a summary string suitable for the report stats.
    """
    weights = get_interest_weights(db_path)
    if not weights:
        return "No interest data yet."

    top = weights[:5]
    parts = [f"{w['interest']}({w['weight']:.1f})" for w in top]
    return "Top interests: " + ", ".join(parts)
