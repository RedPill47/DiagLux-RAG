"""Robust extraction of the chosen letter (A-D) from raw LLM output.

Contract (docs/CONTRACTS.md): ``parsed_letter`` is "A"-"D" or null, and
``parse_status`` is one of ``exact | extracted | unparseable``.

Order of attempts:

1. "exact"      - the stripped output is exactly one letter A-D
                  (case-insensitive, optional trailing punctuation).
2. "extracted"  - an explicit answer marker: "Answer: X" / "the answer is X",
                  bold "**X**", or parenthesised "(X)".
3. "extracted"  - a *standalone* capital A-D token in the output.
4. "unparseable" - nothing found; scored as incorrect downstream but tracked
                  separately via parse_status.

For attempts 2 and 3 the **last** matching occurrence is taken, not the first.
Some models (e.g. Claude Sonnet 4.6) ignore the "return only the letter"
instruction and emit a chain-of-thought that discusses several options
("option (A) is about ..., but the text supports ...") before concluding. The
decision is at the end, so the last marker/letter is the reliable signal; for
terse outputs (a single letter or marker) last == first, so this never hurts.

Standalone means the letter is not adjacent to word characters or
apostrophes. The apostrophe guard matters for Luxembourgish: the definite
article contraction ("D'Kanner", "d'Buch") would otherwise make "D" match
spuriously, and capitals inside words ("Auto", "Den") must never match.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Exactly one letter, optionally wrapped/followed by light punctuation,
# e.g. "B", "b", "D.", "C)", "A:".
_EXACT_RE = re.compile(r"^([A-Da-d])[\s.,;:!)\]]*$")

# Boundary used after a candidate letter: not a word char or apostrophe.
_END = r"(?![\w'’])"
# Boundary used before a candidate letter.
_START = r"(?<![\w'’])"

# "Answer: X", "answer is X", "the answer is **B**." etc.
_ANSWER_RE = re.compile(
    r"\banswer\b(?:\s+is)?\s*[:\-–]?\s*\*{0,2}\(?([A-Da-d])\)?\.?\*{0,2}" + _END,
    re.IGNORECASE,
)

# Bold markdown: **B**, **B.**, **(B)**.
_BOLD_RE = re.compile(r"\*\*\s*\(?([A-Da-d])\)?\.?\s*\*\*")

# Parenthesised: (B) / (b).
_PAREN_RE = re.compile(r"\(\s*([A-Da-d])\s*\)")

# Standalone capital letter token, e.g. "B." in "Et ass B." or an option
# echo "C. Si geet heem." Uppercase only, to avoid Luxembourgish function
# words and mid-sentence lowercase letters.
_STANDALONE_RE = re.compile(_START + r"([A-D])" + _END)

# A letter immediately preceded by a negation ("net C", "not A", "keng B") is a
# *rejected* option, not the answer. Luxembourgish: net/nët/keen/keng/kee;
# English: not/no. Checked against the short text just before the letter.
_NEG_RE = re.compile(r"(?:net|n[eë]t|not|no|keng?|kee[nm]?)\s+$", re.IGNORECASE)


def _last_match(pattern: re.Pattern, text: str):
    """Return the last match of ``pattern`` in ``text`` (or None).

    Conclusions in a chain-of-thought come at the end, so the last occurrence
    of an answer marker is the reliable signal.
    """
    last = None
    for last in pattern.finditer(text):
        pass
    return last


def _best_standalone(text: str):
    """Pick the answer among bare standalone A-D letters.

    Drop letters immediately preceded by a negation ("net C", "not A") since
    those are rejected options, then take the last remaining (a CoT concludes
    at the end). If every candidate is negated, fall back to the last overall.
    """
    matches = list(_STANDALONE_RE.finditer(text))
    if not matches:
        return None
    non_negated = [
        m for m in matches
        if not _NEG_RE.search(text[max(0, m.start() - 8):m.start()])
    ]
    return (non_negated or matches)[-1]


def parse_letter(raw: Optional[str]) -> Tuple[Optional[str], str]:
    """Parse an answer letter out of raw model output.

    Returns ``(letter, parse_status)`` where letter is "A"-"D" or None and
    parse_status is "exact", "extracted", or "unparseable".
    """
    if raw is None:
        return None, "unparseable"
    stripped = raw.strip()
    if not stripped:
        return None, "unparseable"

    m = _EXACT_RE.match(stripped)
    if m:
        return m.group(1).upper(), "exact"

    for pattern in (_ANSWER_RE, _BOLD_RE, _PAREN_RE):
        m = _last_match(pattern, stripped)
        if m:
            return m.group(1).upper(), "extracted"

    m = _best_standalone(stripped)
    if m:
        return m.group(1), "extracted"

    return None, "unparseable"
