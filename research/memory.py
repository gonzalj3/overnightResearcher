"""Memory context builder for synthesis prompts."""

import json

from research.db import get_top_developments, get_interest_weights, get_trending_topics


def build_memory_context(db_path, days=7, max_items=10):
    """Build a concise memory context block from recent DB entries.

    Returns a string suitable for injection into synthesis prompts.
    Targets < 1000 tokens (~4000 chars).
    """
    sections = []

    # Top developments from last week
    top = get_top_developments(db_path, days=days, limit=max_items)
    if top:
        lines = ["Recent top developments:"]
        for item in top:
            title = item.get("title", "Untitled")[:80]
            score = item.get("relevance_score", 0)
            lines.append(f"- {title} (score: {score:.1f})")
        sections.append("\n".join(lines))

    # Interest weights
    weights = get_interest_weights(db_path)
    if weights:
        lines = ["Interest weights:"]
        for w in weights[:10]:
            lines.append(f"- {w['interest']}: {w['weight']:.1f} ({w['total_hits']} hits)")
        sections.append("\n".join(lines))

    # Trending topics
    trending = get_trending_topics(db_path, days=days)
    if trending:
        lines = ["Trending topics:"]
        for t in trending[:10]:
            lines.append(f"- {t['tag']}: {t['count']} mentions")
        sections.append("\n".join(lines))

    context = "\n\n".join(sections)

    # Truncate if too long (~4000 chars ≈ 1000 tokens)
    if len(context) > 3800:
        context = context[:3800] + "\n..."

    return context
