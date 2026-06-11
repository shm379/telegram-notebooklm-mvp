"""Keyword rule matching for auto-tagging ingested content.

A rule maps a keyword to a tag. When a piece of content's text contains a rule's
keyword (case-insensitive substring), the content is tagged. Matching is kept as a
pure function so it is easy to test and reuse across the ingest paths.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def match_tags(text: str | None, rules: Iterable[Mapping[str, str]]) -> set[str]:
    """Return the set of tags whose rule keyword appears in ``text``."""
    lowered = (text or "").lower()
    if not lowered:
        return set()
    matched: set[str] = set()
    for rule in rules:
        keyword = (rule.get("keyword") or "").strip().lower()
        tag = rule.get("tag")
        if keyword and tag and keyword in lowered:
            matched.add(tag)
    return matched
