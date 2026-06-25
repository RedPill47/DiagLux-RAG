"""Inline linguistic annotation tags.

Tags look like ``<LEX-FALSE-FRIEND>`` and follow the word they annotate.
The tag prefixes (LEX / SYN / MORPH / ORTHO / DISC) are the linguistic
categories used in the diagnostic analysis.
"""

from __future__ import annotations

import re

TAG_RE = re.compile(r"<[A-Z][A-Z0-9-]*>")

# Canonical category order for stable, readable output.
CATEGORY_ORDER = ("LEX", "SYN", "MORPH", "ORTHO", "DISC")


def strip_tags(span: str) -> str:
    """Remove all annotation tags from a span (whitespace left untouched)."""
    return TAG_RE.sub("", span)


def extract_tags(span: str) -> list[str]:
    """Return the tags occurring in *span*, deduplicated, first-seen order."""
    seen: dict[str, None] = {}
    for m in TAG_RE.finditer(span):
        seen.setdefault(m.group(0)[1:-1])
    return list(seen)


def tag_categories(tags: list[str]) -> list[str]:
    """Unique tag prefixes (text before the first hyphen), canonical order.

    Unknown prefixes are kept (appended after the known ones) rather than
    dropped, so future annotation schemes surface instead of vanishing.
    """
    prefixes = {tag.split("-", 1)[0] for tag in tags}
    ordered = [c for c in CATEGORY_ORDER if c in prefixes]
    ordered.extend(sorted(p for p in prefixes if p not in CATEGORY_ORDER))
    return ordered
