"""User memory system — extracts and manages facts from iMessage conversations.

Uses Mem0-style CRUD classification (ADD/UPDATE/DELETE/NOOP) to maintain
a persistent user profile that improves research relevance over time.
"""

import json
import logging

from research.db import (
    get_active_facts,
    find_similar_facts,
    insert_user_fact,
    insert_user_message,
    mark_message_processed,
    update_fact,
)
from research.gpu_timesheet import ollama_generate
from research.json_repair import repair_json

logger = logging.getLogger(__name__)

MODEL = "qwen3:8b"  # Use 8B for fast fact extraction (already loaded for chat)


def save_message(db_path, text, sender):
    """Store an incoming user message and return its ID."""
    return insert_user_message(db_path, text, sender)


def extract_facts_from_message(text, existing_facts=None):
    """Extract actionable facts from a user message using LLM.

    Uses Mem0-style CRUD classification:
    - ADD: new fact not in existing memory
    - UPDATE: modifies/replaces an existing fact (include fact_id)
    - DELETE: user explicitly says they no longer care about something (include fact_id)
    - NOOP: message has no extractable preferences/facts

    Returns list of operation dicts:
    [{"operation": "ADD", "fact_type": "interest", "fact_text": "...", "confidence": 0.8}]
    """
    existing_summary = ""
    if existing_facts:
        lines = [f"- [id={f['id']}] ({f['fact_type']}) {f['fact_text']}" for f in existing_facts[:20]]
        existing_summary = "\n".join(lines)

    prompt = f"""/no_think
Extract user preferences, interests, or facts from this message.
Classify each as ADD (new fact), UPDATE (replaces existing fact), DELETE (user no longer cares), or NOOP (no facts).

Fact types: interest, project, preference, context, dislike

{"Existing user facts:" + chr(10) + existing_summary if existing_summary else "No existing facts."}

User message: "{text}"

Return JSON: {{"operations": [{{"operation": "ADD|UPDATE|DELETE|NOOP", "fact_type": "interest|project|preference|context|dislike", "fact_text": "concise fact", "confidence": 0.0-1.0, "fact_id": null}}]}}
If NOOP, return: {{"operations": [{{"operation": "NOOP"}}]}}"""

    try:
        raw = ollama_generate(
            model=MODEL,
            prompt=prompt,
            caller="user_memory",
            system="You extract user preferences from messages. Output valid JSON only.",
            use_json=True,
            timeout=60,
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        result = repair_json(raw)

        if result.get("parse_failed"):
            return [{"operation": "NOOP"}]

        return result.get("operations", [{"operation": "NOOP"}])

    except Exception as e:
        logger.error("Fact extraction failed: %s", e)
        return [{"operation": "NOOP"}]


def apply_fact_operations(db_path, operations, message_id=None):
    """Execute CRUD operations on the user_memory table.

    Returns dict with counts of each operation performed.
    """
    counts = {"added": 0, "updated": 0, "deleted": 0, "noop": 0}

    for op in operations:
        operation = op.get("operation", "NOOP").upper()

        if operation == "NOOP":
            counts["noop"] += 1
            continue

        if operation == "ADD":
            fact_type = op.get("fact_type", "context")
            fact_text = op.get("fact_text", "")
            confidence = op.get("confidence", 0.8)
            if fact_text:
                # Check for duplicates before adding
                similar = find_similar_facts(db_path, fact_text, fact_type)
                if not similar:
                    insert_user_fact(db_path, fact_type, fact_text, confidence, message_id)
                    counts["added"] += 1
                else:
                    logger.debug("Skipping duplicate fact: %s", fact_text[:50])

        elif operation == "UPDATE":
            fact_id = op.get("fact_id")
            fact_text = op.get("fact_text", "")
            confidence = op.get("confidence")
            if fact_id and fact_text:
                update_fact(db_path, fact_id, fact_text=fact_text, confidence=confidence)
                counts["updated"] += 1

        elif operation == "DELETE":
            fact_id = op.get("fact_id")
            if fact_id:
                update_fact(db_path, fact_id, status="deleted")
                counts["deleted"] += 1

    return counts


def process_message(db_path, text, sender):
    """Full pipeline: save message, extract facts, apply operations.

    Returns dict with message_id and operation counts.
    """
    # Save the message
    message_id = save_message(db_path, text, sender)

    # Get existing facts for context
    existing_facts = get_active_facts(db_path)

    # Extract facts from the message
    operations = extract_facts_from_message(text, existing_facts)

    # Apply operations
    counts = apply_fact_operations(db_path, operations, message_id)

    # Mark message as processed
    mark_message_processed(db_path, message_id, extracted_facts=operations)

    logger.info(
        "Processed message %d: added=%d, updated=%d, deleted=%d, noop=%d",
        message_id, counts["added"], counts["updated"], counts["deleted"], counts["noop"],
    )

    return {"message_id": message_id, **counts}


def build_user_profile(db_path):
    """Format active facts as a <user_profile> XML block for prompt injection.

    Returns string (<800 chars) or empty string if no facts.
    """
    facts = get_active_facts(db_path)
    if not facts:
        return ""

    lines = ["<user_profile>"]
    char_count = len(lines[0])

    # Group by fact_type
    by_type = {}
    for f in facts:
        ft = f.get("fact_type", "context")
        by_type.setdefault(ft, []).append(f["fact_text"])

    for fact_type, texts in by_type.items():
        header = f"  {fact_type}:"
        if char_count + len(header) > 750:
            break
        lines.append(header)
        char_count += len(header)
        for text in texts:
            entry = f"    - {text}"
            if char_count + len(entry) > 750:
                break
            lines.append(entry)
            char_count += len(entry)

    lines.append("</user_profile>")
    return "\n".join(lines)


def build_focus_boost_tags(db_path):
    """Extract topic tags from 'interest' and 'project' facts for synthesis boosting.

    Returns list of tag strings.
    """
    facts = get_active_facts(db_path)
    tags = []
    for f in facts:
        if f.get("fact_type") in ("interest", "project"):
            # Use the fact text as a tag (it's already concise)
            tags.append(f["fact_text"])
    return tags[:10]  # Cap at 10 focus tags
