"""Generic HTTP fetcher with optional CSS selector extraction."""

import requests
from bs4 import BeautifulSoup
import html2text

TIMEOUT = 10


def fetch_url(url, selector=None):
    """Fetch a URL and convert to markdown.

    Args:
        url: URL to fetch
        selector: Optional CSS selector to extract specific elements

    Returns markdown string, or None on error.
    """
    try:
        resp = requests.get(
            url,
            timeout=TIMEOUT,
            headers={'User-Agent': 'OvernightResearcher/1.0'}
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return None

    try:
        if selector:
            soup = BeautifulSoup(html, 'lxml')
            elements = soup.select(selector)
            if not elements:
                return None
            html = '\n'.join(str(el) for el in elements)

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0
        return converter.handle(html)
    except Exception:
        return None
