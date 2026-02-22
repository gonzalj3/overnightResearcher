"""GitHub README fetcher."""

import requests

TIMEOUT = 10


def fetch_readme(repo):
    """Fetch raw README for a GitHub repo.

    Args:
        repo: "owner/name" string

    Returns markdown string, or None on error.
    """
    try:
        resp = requests.get(
            f'https://api.github.com/repos/{repo}/readme',
            headers={'Accept': 'application/vnd.github.raw+json'},
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None
