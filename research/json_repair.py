"""JSON repair utility for malformed LLM output."""

import json
import re


def repair_json(text):
    """Attempt to parse JSON from potentially malformed LLM output.

    Strategies (in order):
    1. Direct parse
    2. Strip markdown code fences
    3. Fix trailing commas
    4. Fix single quotes to double quotes
    5. Extract JSON-like substring with regex
    6. Return parse_failed sentinel
    """
    if isinstance(text, dict):
        return text

    if not isinstance(text, str) or not text.strip():
        return _sentinel("empty input")

    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown code fences
    stripped = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    stripped = re.sub(r'\n?```\s*$', '', stripped, flags=re.MULTILINE)
    stripped = stripped.strip()
    if stripped != text:
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            text = stripped

    # 3. Fix trailing commas (before } or ])
    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # 4. Fix single quotes to double quotes
    sq_fixed = text.replace("'", '"')
    sq_fixed = re.sub(r',\s*([}\]])', r'\1', sq_fixed)
    try:
        return json.loads(sq_fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # 5. Extract JSON-like substring with regex
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            # Try with single-quote fix too
            try:
                return json.loads(candidate.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                pass

    # 6. Final fallback
    return _sentinel(f"could not parse: {text[:200]}")


def _sentinel(reason):
    return {
        "parse_failed": True,
        "title": "Parse failed",
        "summary": reason,
        "relevance_tags": [],
        "relevance_score": 0.0,
        "key_developments": [],
    }
