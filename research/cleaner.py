"""Content cleaning and truncation."""

import re


def clean_content(text, max_chars=2000):
    """Strip excess whitespace and truncate at word boundary."""
    if not text:
        return ""
    # Normalize whitespace: collapse runs of whitespace to single space/newline
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Truncate at last word boundary before max_chars
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + '...'
