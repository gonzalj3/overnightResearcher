"""Main fetch pipeline orchestrator."""

import hashlib
import json
import os
from datetime import datetime, timezone

import yaml

from research.fetch_hn import fetch_hn_stories
from research.fetch_github import fetch_readme
from research.fetch_rss import fetch_rss
from research.fetch_http import fetch_url
from research.fetch_nitter import fetch_tweets
from research.cleaner import clean_content


def run_fetch_pipeline(sources_path='research/sources.yaml', max_total=None):
    """Run the full fetch pipeline.

    1. Load sources.yaml
    2. Call each fetch method based on source type
    3. Clean each result
    4. Save to research/raw/{date}/
    5. Write manifest.json

    Returns path to the output directory.
    """
    with open(sources_path) as f:
        config = yaml.safe_load(f)

    sources = config.get('sources', {})
    raw_dir = config.get('output', {}).get('raw_dir', 'research/raw')
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    out_dir = os.path.join(raw_dir, date_str)
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    index = 0
    total = 0

    # Hacker News
    hn_config = sources.get('hacker_news', {})
    if hn_config.get('enabled', False):
        stories = fetch_hn_stories(max_items=hn_config.get('max_items', 50))
        for story in stories:
            if max_total and total >= max_total:
                break
            content = f"# {story['title']}\n\nScore: {story['score']}\nURL: {story['url']}\nComments: {story['comments_url']}"
            content = clean_content(content)
            fname = f"hn_{index}.md"
            _write_item(out_dir, fname, content)
            manifest.append(_manifest_entry(
                story.get('url', ''), 'hn', story.get('title', ''), content
            ))
            index += 1
            total += 1

    # GitHub READMEs
    gh_config = sources.get('github_repos', {})
    if gh_config.get('enabled', False):
        for repo in gh_config.get('repos', []):
            if max_total and total >= max_total:
                break
            readme_text = fetch_readme(repo)
            if not readme_text:
                continue
            content = clean_content(readme_text)
            fname = f"github_{index}.md"
            _write_item(out_dir, fname, content)
            manifest.append(_manifest_entry(
                f'https://github.com/{repo}', 'github', repo, content
            ))
            index += 1
            total += 1

    # RSS feeds
    rss_config = sources.get('rss_feeds', {})
    if rss_config.get('enabled', False):
        for feed_info in rss_config.get('feeds', []):
            if max_total and total >= max_total:
                break
            items = fetch_rss(
                feed_info['url'],
                max_items=rss_config.get('max_items', 100)
            )
            for item in items:
                if max_total and total >= max_total:
                    break
                content = f"# {item['title']}\n\n{item['content']}"
                content = clean_content(content)
                fname = f"rss_{index}.md"
                _write_item(out_dir, fname, content)
                manifest.append(_manifest_entry(
                    item.get('url', ''), 'rss', item.get('title', ''), content
                ))
                index += 1
                total += 1

    # Websites
    http_config = sources.get('websites', {})
    if http_config.get('enabled', False):
        for url_info in http_config.get('urls', []):
            if max_total and total >= max_total:
                break
            text = fetch_url(url_info['url'], selector=url_info.get('selector'))
            if not text:
                continue
            content = clean_content(text)
            fname = f"http_{index}.md"
            _write_item(out_dir, fname, content)
            manifest.append(_manifest_entry(
                url_info['url'], 'http', url_info.get('name', ''), content
            ))
            index += 1
            total += 1

    # Twitter/Nitter
    nitter_config = sources.get('twitter', {})
    if nitter_config.get('enabled', False):
        tweet_list = fetch_tweets(
            nitter_config.get('accounts', []),
            nitter_config.get('proxy_url', ''),
            max_items=nitter_config.get('max_items', 200)
        )
        for tweet in tweet_list:
            if max_total and total >= max_total:
                break
            content = f"@{tweet['author']}: {tweet['text']}"
            content = clean_content(content)
            fname = f"nitter_{index}.md"
            _write_item(out_dir, fname, content)
            manifest.append(_manifest_entry(
                tweet.get('url', ''), 'nitter', f"@{tweet['author']}", content
            ))
            index += 1
            total += 1

    # Write manifest
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return out_dir


def _write_item(out_dir, fname, content):
    path = os.path.join(out_dir, fname)
    with open(path, 'w') as f:
        f.write(content)


def _manifest_entry(url, source_type, title, content):
    return {
        'url': url,
        'source_type': source_type,
        'title': title,
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'content_hash': hashlib.sha256(content.encode()).hexdigest(),
    }
