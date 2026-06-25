"""Chunking strategies over the clean text body.

All offsets are character offsets into the clean body exactly as returned
by ``load_clean_text`` (the same coordinate system as the span offsets),
and ``chunk_text == body[start_char:end_char]`` always holds.

Strategies (docs/CONTRACTS.md):

- ``paragraph``: natural units. The clean texts contain no blank lines
  (they are hard-wrapped with single newlines), so the blank-line split is
  attempted first and, when it yields a single oversized unit, the text is
  regrouped into natural units: consecutive lines accumulated until a line
  ends with sentence-final punctuation and the unit holds >= 60 tokens
  (hard cap 180 tokens for punctuation-poor passages).
- ``overlap``: ~150 whitespace-token windows with 50% overlap; the union of
  chunk spans covers the body completely (asserted via
  ``check_full_coverage``).
- ``sentence``: simple rule-based splitting at sentence-final punctuation
  (``. ! ? …`` plus trailing closing quotes/brackets) followed by
  whitespace, and at line breaks that end with such punctuation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\S+")
BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
# Sentence-final punctuation, optionally followed by closing quotes/brackets.
SENT_END_RE = re.compile(r"[.!?…]+[»«\"'„“”’)\]]*")
SENT_END_AT_EOL_RE = re.compile(r"[.!?…][»«\"'„“”’)\]]*$")

PARA_MIN_TOKENS = 60
PARA_MAX_TOKENS = 180
OVERLAP_WINDOW = 150
OVERLAP_FRACTION = 0.5


@dataclass
class Chunk:
    chunk_id: str
    text_id: str
    chunk_text: str
    start_char: int
    end_char: int
    n_tokens: int


def _n_tokens(s: str) -> int:
    return len(s.split())


def _trim(body: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it does not begin or end in whitespace."""
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return start, end


def _build(text_id: str, strategy: str, body: str, spans: list[tuple[int, int]]) -> list[Chunk]:
    chunks = []
    for idx, (start, end) in enumerate(spans):
        text = body[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{text_id}_{strategy}_c{idx:03d}",
                text_id=text_id,
                chunk_text=text,
                start_char=start,
                end_char=end,
                n_tokens=_n_tokens(text),
            )
        )
    return chunks


# --------------------------------------------------------------------------
# paragraph
# --------------------------------------------------------------------------

def _blank_line_spans(body: str) -> list[tuple[int, int]]:
    spans, last = [], 0
    for m in BLANK_LINE_RE.finditer(body):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(body)))
    out = []
    for s, e in spans:
        s, e = _trim(body, s, e)
        if s < e:
            out.append((s, e))
    return out


def _line_group_spans(
    body: str,
    start: int,
    end: int,
    min_tokens: int = PARA_MIN_TOKENS,
    max_tokens: int = PARA_MAX_TOKENS,
) -> list[tuple[int, int]]:
    """Group hard-wrapped lines into natural units.

    A unit is flushed after a line whose content ends with sentence-final
    punctuation once it holds >= min_tokens, or unconditionally once it
    exceeds max_tokens (free verse / punctuation-poor passages).
    """
    segment = body[start:end]
    spans: list[tuple[int, int]] = []
    unit_start: int | None = None
    unit_tokens = 0
    pos = 0
    for line in segment.splitlines(keepends=True):
        line_start, line_end = pos, pos + len(line)
        pos = line_end
        content = line.rstrip("\n")
        if content.strip():
            if unit_start is None:
                unit_start = line_start
            unit_tokens += _n_tokens(content)
            ends_sentence = bool(SENT_END_AT_EOL_RE.search(content.rstrip()))
            if (ends_sentence and unit_tokens >= min_tokens) or unit_tokens >= max_tokens:
                spans.append((start + unit_start, start + line_start + len(content)))
                unit_start, unit_tokens = None, 0
    if unit_start is not None:
        spans.append((start + unit_start, end))
    return [
        (s2, e2)
        for s, e in spans
        for s2, e2 in [_trim(body, s, e)]
        if s2 < e2
    ]


def chunk_paragraph(text_id: str, body: str) -> list[Chunk]:
    spans: list[tuple[int, int]] = []
    for s, e in _blank_line_spans(body):
        if _n_tokens(body[s:e]) > PARA_MAX_TOKENS:
            spans.extend(_line_group_spans(body, s, e))
        else:
            spans.append((s, e))
    return _build(text_id, "paragraph", body, spans)


# --------------------------------------------------------------------------
# overlap
# --------------------------------------------------------------------------

def chunk_overlap(
    text_id: str,
    body: str,
    window: int = OVERLAP_WINDOW,
    overlap: float = OVERLAP_FRACTION,
) -> list[Chunk]:
    tokens = list(TOKEN_RE.finditer(body))
    if not tokens:
        return []
    step = max(1, window - int(window * overlap))
    bounds: list[tuple[int, int]] = []  # token-index windows
    i = 0
    while True:
        j = min(i + window, len(tokens))
        bounds.append((i, j))
        if j >= len(tokens):
            break
        i += step
    spans = []
    for k, (i, j) in enumerate(bounds):
        start = 0 if k == 0 else tokens[i].start()
        end = len(body) if k == len(bounds) - 1 else tokens[j - 1].end()
        spans.append((start, end))
    chunks = _build(text_id, "overlap", body, spans)
    check_full_coverage(chunks, len(body))
    return chunks


def check_full_coverage(chunks: list[Chunk], body_len: int) -> None:
    """Assert that the union of chunk spans covers [0, body_len) completely."""
    if body_len == 0:
        return
    if not chunks:
        raise AssertionError("no chunks for non-empty body")
    spans = sorted((c.start_char, c.end_char) for c in chunks)
    if spans[0][0] != 0:
        raise AssertionError(f"coverage gap at start: first chunk begins at {spans[0][0]}")
    reach = spans[0][1]
    for s, e in spans[1:]:
        if s > reach:
            raise AssertionError(f"coverage gap: [{reach}, {s})")
        reach = max(reach, e)
    if reach != body_len:
        raise AssertionError(f"coverage gap at end: {reach} != {body_len}")


# --------------------------------------------------------------------------
# sentence
# --------------------------------------------------------------------------

def chunk_sentence(text_id: str, body: str) -> list[Chunk]:
    """Rule-based sentence splitting with clean-body offsets."""
    cut_points = {0, len(body)}
    for m in SENT_END_RE.finditer(body):
        nxt = m.end()
        # Split only when followed by whitespace (or end of text); avoids
        # splitting inside "1000-mol", abbreviations glued to words, etc.
        if nxt >= len(body) or body[nxt].isspace():
            cut_points.add(nxt)
    # Also split at blank lines so verse/paragraph breaks separate units.
    for m in BLANK_LINE_RE.finditer(body):
        cut_points.add(m.start())
    cuts = sorted(cut_points)
    spans = []
    for s, e in zip(cuts, cuts[1:]):
        s, e = _trim(body, s, e)
        if s < e:
            spans.append((s, e))
    return _build(text_id, "sentence", body, spans)
