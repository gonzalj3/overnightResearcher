"""Hacker News API fetcher."""

import requests

TIMEOUT = 10


def fetch_hn_stories(max_items=50):
    """Fetch top stories from HN Firebase API.

    Returns list of {title, url, score, comments_url}.
    """
    try:
        resp = requests.get(
            'https://hacker-news.firebaseio.com/v0/topstories.json',
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        story_ids = resp.json()[:max_items]
    except Exception:
        return []

    stories = []
    for sid in story_ids:
        try:
            resp = requests.get(
                f'https://hacker-news.firebaseio.com/v0/item/{sid}.json',
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            item = resp.json()
            if not item:
                continue
            stories.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'score': item.get('score', 0),
                'comments_url': f'https://news.ycombinator.com/item?id={sid}',
            })
        except Exception:
            continue
    return stories
