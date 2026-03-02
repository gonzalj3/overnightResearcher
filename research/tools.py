"""Tool registry, executor, and agentic loop for LLM tool calling."""

import json
import logging

from research.cleaner import clean_content
from research.gpu_timesheet import ollama_chat

logger = logging.getLogger(__name__)

MODEL = "qwen3:8b"

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and return its content as markdown text",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_hacker_news",
            "description": "Get the current top stories from Hacker News",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of stories to fetch (default 10)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_github_readme",
            "description": "Fetch the README of a GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in owner/name format, e.g. 'ollama/ollama'",
                    },
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_research_db",
            "description": "Search recent research summaries from the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "How many days back to search (default 7)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending_topics",
            "description": "Get trending topics from the research database",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "How many days to look back (default 14)",
                    },
                },
                "required": [],
            },
        },
    },
]

# Set of valid tool names for client-side validation (Ollama can hallucinate names)
_VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOL_DEFINITIONS}


def execute_tool(name, arguments, db_path="research/research.db"):
    """Execute a tool by name and return the result as a string.

    Args:
        name: Tool function name
        arguments: Dict of arguments
        db_path: Path to SQLite database

    Returns:
        Result string, truncated to 2000 chars.
    """
    if name not in _VALID_TOOL_NAMES:
        return f"Error: unknown tool '{name}'"

    try:
        if name == "fetch_url":
            from research.fetch_http import fetch_url
            content = fetch_url(arguments["url"])
            if not content:
                return f"Could not fetch {arguments['url']}"
            return clean_content(content, max_chars=2000)

        elif name == "fetch_hacker_news":
            from research.fetch_hn import fetch_hn_stories
            count = arguments.get("count", 10)
            stories = fetch_hn_stories(min(count, 30))
            if not stories:
                return "Could not reach Hacker News"
            lines = []
            for s in stories[:count]:
                lines.append(
                    f"- {s['title']} ({s['score']} pts) {s['url']}"
                )
            return "\n".join(lines)

        elif name == "fetch_github_readme":
            from research.fetch_github import fetch_readme
            content = fetch_readme(arguments["repo"])
            if not content:
                return f"Could not fetch README for {arguments['repo']}"
            return clean_content(content, max_chars=2000)

        elif name == "search_research_db":
            from research.db import get_recent_sources
            days = arguments.get("days", 7)
            limit = arguments.get("limit", 10)
            sources = get_recent_sources(db_path, days=days)
            sources.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            sources = sources[:limit]
            if not sources:
                return "No recent research found."
            lines = []
            for s in sources:
                title = s.get("title", "Untitled")[:80]
                summary = s.get("summary", "")[:150]
                score = s.get("relevance_score", 0)
                lines.append(f"- [{score:.1f}] {title}: {summary}")
            return "\n".join(lines)

        elif name == "get_trending_topics":
            from research.db import get_trending_topics
            days = arguments.get("days", 14)
            topics = get_trending_topics(db_path, days=days)
            if not topics:
                return "No trending topics found."
            lines = []
            for t in topics[:15]:
                lines.append(f"- {t['tag']} ({t['count']} mentions)")
            return "\n".join(lines)

    except Exception as e:
        logger.error("Tool '%s' failed: %s", name, e)
        return f"Error executing {name}: {e}"

    return f"Error: unhandled tool '{name}'"


def run_tool_loop(model, messages, caller, db_path="research/research.db",
                  max_iterations=3):
    """Run the agentic tool-calling loop.

    1. Call ollama_chat() with tools
    2. If response has tool_calls, execute each, append results
    3. Call model again with updated messages
    4. Repeat until no more tool_calls or max iterations hit

    Args:
        model: Ollama model name
        messages: Initial message list [{role, content}, ...]
        caller: Timesheet identifier
        db_path: Path to SQLite database
        max_iterations: Max tool-call round-trips

    Returns:
        Final text response string, or None on failure.
    """
    options = {"temperature": 0.1, "num_ctx": 8192}

    for i in range(max_iterations):
        try:
            response = ollama_chat(
                model=model,
                messages=messages,
                caller=f"{caller}.loop_{i}",
                tools=TOOL_DEFINITIONS,
                timeout=120,
                options=options,
            )
        except Exception as e:
            logger.error("ollama_chat failed on iteration %d: %s", i, e)
            return None

        tool_calls = response.get("tool_calls")
        content = response.get("content")

        # No tool calls — model gave a final answer
        if not tool_calls:
            return (content or "").strip() or None

        # Append the assistant message (with tool_calls) to history
        messages.append(response)

        # Execute each tool call and append results
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            tool_args = func.get("arguments", {})

            # Client-side validation: skip hallucinated tool names
            if tool_name not in _VALID_TOOL_NAMES:
                logger.warning("Skipping hallucinated tool: %s", tool_name)
                result = f"Error: unknown tool '{tool_name}'"
            else:
                logger.info("Executing tool: %s(%s)", tool_name, tool_args)
                result = execute_tool(tool_name, tool_args, db_path=db_path)

            messages.append({
                "role": "tool",
                "content": result,
            })

    # Exhausted iterations — do one final call without tools to get a text reply
    try:
        response = ollama_chat(
            model=model,
            messages=messages,
            caller=f"{caller}.final",
            timeout=120,
            options=options,
        )
        return (response.get("content") or "").strip() or None
    except Exception as e:
        logger.error("Final ollama_chat failed: %s", e)
        return None
