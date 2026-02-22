"""Nitter RSS proxy fetcher for Twitter/X accounts."""

import feedparser
import requests

TIMEOUT = 10


def fetch_tweets(accounts, proxy_url, max_items=200):
    """Fetch tweets via Nitter RSS proxy.

    Args:
        accounts: list of Twitter usernames (without @)
        proxy_url: Nitter instance base URL
        max_items: max total tweets to return

    Returns list of {text, author, url, published}.
    """
    tweets = []
    per_account = max(1, max_items // max(1, len(accounts)))

    for account in accounts:
        feed_url = f'{proxy_url.rstrip("/")}/{account}/rss'
        try:
            resp = requests.get(
                feed_url,
                timeout=TIMEOUT,
                headers={'User-Agent': 'OvernightResearcher/1.0'}
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception:
            continue

        for entry in feed.entries[:per_account]:
            tweets.append({
                'text': getattr(entry, 'title', '') or getattr(entry, 'summary', ''),
                'author': account,
                'url': getattr(entry, 'link', ''),
                'published': getattr(entry, 'published', ''),
            })

        if len(tweets) >= max_items:
            break

    return tweets[:max_items]
