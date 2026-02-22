"""Chunk 2 tests: Source Registry & Fetch Layer.

Tests hit live APIs (HN, GitHub, httpbin, hnrss.org).
Nitter test is lenient (may return 0 results if proxy is down).
"""

import json
import os
import shutil
import tempfile

import pytest
import yaml

from research.fetch_hn import fetch_hn_stories
from research.fetch_github import fetch_readme
from research.fetch_rss import fetch_rss
from research.fetch_http import fetch_url
from research.fetch_nitter import fetch_tweets
from research.cleaner import clean_content
from research.fetcher import run_fetch_pipeline


# TEST 2.1: sources.yaml parses without error and has all 5 source types
def test_sources_yaml_valid():
    with open('research/sources.yaml') as f:
        config = yaml.safe_load(f)
    sources = config['sources']
    assert 'hacker_news' in sources
    assert 'github_repos' in sources
    assert 'rss_feeds' in sources
    assert 'websites' in sources
    assert 'twitter' in sources
    assert sources['hacker_news']['type'] == 'hn'
    assert sources['github_repos']['type'] == 'github'
    assert sources['rss_feeds']['type'] == 'rss'
    assert sources['websites']['type'] == 'http'
    assert sources['twitter']['type'] == 'nitter'


# TEST 2.2: HN fetcher returns >= 10 stories with required fields
def test_hn_fetcher():
    stories = fetch_hn_stories(max_items=15)
    assert len(stories) >= 10
    for story in stories:
        assert 'title' in story
        assert 'url' in story
        assert 'score' in story
        assert 'comments_url' in story
        assert isinstance(story['score'], (int, float))


# TEST 2.3: GitHub README fetcher returns content for a known repo
def test_github_readme():
    readme = fetch_readme('ollama/ollama')
    assert readme is not None
    assert len(readme) > 100
    assert 'ollama' in readme.lower() or 'Ollama' in readme


# TEST 2.4: GitHub fetcher returns None for nonexistent repo
def test_github_readme_nonexistent():
    readme = fetch_readme('thisrepo/definitelydoesnotexist999999')
    assert readme is None


# TEST 2.5: RSS fetcher returns items from hnrss.org
def test_rss_fetcher():
    items = fetch_rss('https://hnrss.org/newest?points=100', max_items=10)
    assert len(items) >= 1
    for item in items:
        assert 'title' in item
        assert 'url' in item
        assert item['title']  # non-empty


# TEST 2.6: HTTP fetcher returns content from httpbin
def test_http_fetcher():
    content = fetch_url('https://httpbin.org/html')
    assert content is not None
    assert len(content) > 50
    # httpbin.org/html returns a page with "Herman Melville"
    assert 'Herman Melville' in content or 'Moby' in content


# TEST 2.7: HTTP fetcher returns None on bad URL
def test_http_fetcher_bad_url():
    content = fetch_url('https://httpbin.org/status/404')
    assert content is None


# TEST 2.8: Cleaner truncates long content at word boundary
def test_cleaner_truncation():
    long_text = 'word ' * 1000  # 5000 chars
    result = clean_content(long_text, max_chars=100)
    assert len(result) <= 104  # 100 + "..."
    assert result.endswith('...')
    # Should not cut mid-word
    assert '  ' not in result


# TEST 2.9: Nitter fetcher handles gracefully (lenient — proxy may be down)
def test_nitter_fetcher():
    tweets = fetch_tweets(
        ['ylecun'],
        'https://nitter.privacydev.net',
        max_items=10
    )
    # Lenient: just verify it returns a list and doesn't crash
    assert isinstance(tweets, list)
    for tweet in tweets:
        assert 'text' in tweet
        assert 'author' in tweet


# TEST 2.10: Full pipeline writes manifest.json with correct structure
def test_full_pipeline_manifest():
    # Use a temp sources.yaml with only HN (fastest source)
    tmpdir = tempfile.mkdtemp()
    try:
        sources = {
            'sources': {
                'hacker_news': {
                    'type': 'hn',
                    'max_items': 3,
                    'enabled': True,
                },
            },
            'output': {
                'raw_dir': os.path.join(tmpdir, 'raw'),
            },
        }
        sources_path = os.path.join(tmpdir, 'sources.yaml')
        with open(sources_path, 'w') as f:
            yaml.dump(sources, f)

        out_dir = run_fetch_pipeline(sources_path, max_total=3)
        assert os.path.isdir(out_dir)

        manifest_path = os.path.join(out_dir, 'manifest.json')
        assert os.path.isfile(manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert len(manifest) >= 1
        for entry in manifest:
            assert 'url' in entry
            assert 'source_type' in entry
            assert 'title' in entry
            assert 'fetched_at' in entry
            assert 'content_hash' in entry
            assert len(entry['content_hash']) == 64  # SHA256 hex
    finally:
        shutil.rmtree(tmpdir)
