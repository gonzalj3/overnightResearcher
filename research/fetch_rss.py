"""RSS/Atom feed fetcher."""

import feedparser
import requests

TIMEOUT = 10


def fetch_rss(feed_url, max_items=100):
    """Parse an RSS/Atom feed.

    Returns list of {title, url, content, published}.
    """
    try:
        # Use requests for HTTP (has proper SSL certs via certifi)
        # then parse the content with feedparser
        resp = requests.get(
            feed_url,
            timeout=TIMEOUT,
            headers={'User-Agent': 'OvernightResearcher/1.0'}
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception:
        return []

    items = []
    for entry in feed.entries[:max_items]:
        content = ''
        if hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'content') and entry.content:
            content = entry.content[0].get('value', '')

        items.append({
            'title': getattr(entry, 'title', ''),
            'url': getattr(entry, 'link', ''),
            'content': content,
            'published': getattr(entry, 'published', ''),
        })
    return items
