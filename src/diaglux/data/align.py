"""Span alignment: locate criticalSpan/distractorSpan in the clean text body.

Spans are substrings of the *annotated* texts (inline tags, merged words,
line-break hyphenation), and the annotated files start with the title and
author inline — several spans therefore begin inside the title/author
prefix. Matching is done whitespace-insensitively against the full page
(title + author + body) and the matched region is clipped to the body:

1. strip tags, squash ALL whitespace from both span and page while keeping
   an index map back to original offsets, substring-match the squashed
   strings -> status "exact" (unique) or "multiple" (several occurrences;
   the first one intersecting the body is recorded, with n_matches);
2. fallback: additionally remove hyphens from both sides (handles
   line-break hyphenation such as "Telefons- buch") -> "dehyphen";
3. fallback: if the span begins with the known title (or title + author)
   of its text, strip that prefix (plus any separating period) from the
   SPAN and re-run steps 1-2 and 4 on the remainder; the inner status is
   kept, so a remainder that matches exactly is still "exact";
4. fallback: fuzzy match — anchor a window with
   difflib.SequenceMatcher.find_longest_match on the squashed,
   dehyphenated page, refine the window position by ratio, and accept it
   if ratio >= the acceptance threshold -> "fuzzy". The threshold is
   FUZZY_THRESHOLD (0.85) for short spans and LONG_FUZZY_THRESHOLD (0.80)
   for spans of >= LONG_SPAN_MIN_SQ (60) squashed characters, where a
   0.80-similar window is still unambiguous;
5. fallback: some annotated spans CONCATENATE non-contiguous passages of
   the text, so the span as a whole matches nowhere. Split the span at
   sentence boundaries (and at spaced dashes / "[…]" omission markers) and
   try to align each piece of >= PARTIAL_MIN_SQ (30) squashed characters,
   longest first, with steps 1-2 and 4. The first piece that aligns
   uniquely is recorded with status "fuzzy" and ``partial=True`` (an
   additive span-schema extension; consumers ignore unknown keys) — the
   offsets then cover only that piece, not the full annotated span;
6. otherwise "unresolved". An empty span (after tag stripping/squashing)
   is "empty".

If a located span lies entirely inside the title/author prefix (two
criticalSpans equal the story title), offsets are None and in_title=True,
because the clean body excludes the title and author lines. Otherwise the
returned offsets always index into the clean text body exactly as produced
by ``diaglux.data.texts.load_clean_text`` (docs/CONTRACTS.md), clipped to
the body when a match starts inside the prefix.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from diaglux.data.tags import strip_tags

FUZZY_THRESHOLD = 0.85
# Long spans may accept a slightly lower ratio: at >= 60 squashed chars a
# 0.80-similar window is still unambiguous, while short spans keep the
# stricter threshold so precision does not degrade.
LONG_FUZZY_THRESHOLD = 0.80
LONG_SPAN_MIN_SQ = 60
# Minimum squashed length of a sentence piece considered in the partial
# (concatenated-span) fallback; shorter pieces match too promiscuously.
PARTIAL_MIN_SQ = 30
_HYPHENS = "-‐‑"

# Sentence-final punctuation run, optionally followed by a closing quote.
_SENT_END_RE = re.compile(r'[.!?…]+["”“»«]?')
# Hard separators inside concatenated spans: spaced en/em dashes and
# "[…]" / "[...]" omission markers.
_SEPARATOR_RE = re.compile(r"\s[–—]\s|\[…\]|\[\.\.\.\]")
_OPENING_QUOTES = '„«"'


@dataclass
class SpanAlignment:
    start: int | None
    end: int | None
    status: str  # exact | dehyphen | fuzzy | multiple | unresolved | empty
    in_title: bool = False
    ratio: float | None = None  # similarity of the accepted fuzzy window
    n_matches: int | None = None  # occurrence count when status == "multiple"
    partial: bool = False  # offsets cover one piece of a concatenated span


@dataclass
class _Page:
    """Squashed views of the match page (title/author prefix + body)."""

    sq: str
    sq_map: list[int]
    dh: str
    dh_map: list[int]
    body_offset: int


def _squash(s: str, *, drop_hyphens: bool = False) -> tuple[str, list[int]]:
    """Remove all whitespace (and optionally hyphens); map kept chars to
    their original indices."""
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(s):
        if ch.isspace():
            continue
        if drop_hyphens and ch in _HYPHENS:
            continue
        chars.append(ch)
        index_map.append(i)
    return "".join(chars), index_map


def _find_all(needle: str, hay: str) -> list[int]:
    hits = []
    pos = hay.find(needle)
    while pos != -1:
        hits.append(pos)
        pos = hay.find(needle, pos + 1)
    return hits


def _clip_to_body(
    index_map: list[int],
    sq_start: int,
    sq_end: int,
    body_offset: int,
    *,
    status: str,
    ratio: float | None = None,
    n_matches: int | None = None,
) -> SpanAlignment:
    """Squashed page window -> body-coordinate SpanAlignment.

    Positions before *body_offset* belong to the title/author prefix; a
    match is clipped to its body part, and a match entirely inside the
    prefix yields offsets None with in_title=True.
    """
    # First squashed position of the window that lies inside the body.
    first_in_body = None
    for i in range(sq_start, sq_end):
        if index_map[i] >= body_offset:
            first_in_body = i
            break
    if first_in_body is None:
        return SpanAlignment(
            start=None, end=None, status=status, in_title=True,
            ratio=ratio, n_matches=n_matches,
        )
    start = index_map[first_in_body] - body_offset
    end = index_map[sq_end - 1] + 1 - body_offset
    return SpanAlignment(start=start, end=end, status=status, ratio=ratio, n_matches=n_matches)


def _pick_hit(hits: list[int], length: int, index_map: list[int], body_offset: int) -> int:
    """Prefer the first occurrence that intersects the body."""
    for h in hits:
        if index_map[h + length - 1] >= body_offset:
            return h
    return hits[0]


def _fuzzy_locate(
    span_sq: str, page_sq: str, threshold: float
) -> tuple[int, int, float] | None:
    """Best approximately matching window of *page_sq* for *span_sq*.

    Returns (start, end, ratio) in squashed-page coordinates, or None if no
    window reaches *threshold*. The window is anchored on the longest common
    substring (robust against the near-uniform character distributions of
    running text, where multiset-based heuristics are uninformative),
    refined by ratio in a small neighbourhood, then trimmed to its
    outermost matching blocks so offsets do not include unmatched fringe.
    """
    n = len(span_sq)
    if n == 0 or not page_sq:
        return None
    anchor_sm = difflib.SequenceMatcher(None, page_sq, span_sq, autojunk=False)
    longest = anchor_sm.find_longest_match(0, len(page_sq), 0, n)
    if longest.size == 0:
        return None
    anchor = longest.a - longest.b  # window start that aligns the common block
    last = max(0, len(page_sq) - n)
    delta = max(8, n // 8)
    lo = max(0, anchor - delta)
    hi = min(last, anchor + delta)
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(span_sq)  # seq2 is cached by SequenceMatcher
    best_ratio, best_start = -1.0, None
    for start in range(lo, hi + 1):
        sm.set_seq1(page_sq[start : start + n])
        if sm.real_quick_ratio() <= best_ratio or sm.quick_ratio() <= best_ratio:
            continue
        r = sm.ratio()
        if r > best_ratio:
            best_ratio, best_start = r, start
    if best_start is None or best_ratio < threshold:
        return None
    window = page_sq[best_start : best_start + n]
    sm.set_seq1(window)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    w_start = blocks[0].a
    w_end = blocks[-1].a + blocks[-1].size
    return best_start + w_start, best_start + w_end, best_ratio


def _build_page(body: str, title: str | None, author: str | None) -> _Page:
    prefix = "".join(p + "\n" for p in (title, author) if p)
    page = prefix + body
    sq, sq_map = _squash(page)
    dh, dh_map = _squash(page, drop_hyphens=True)
    return _Page(sq=sq, sq_map=sq_map, dh=dh, dh_map=dh_map, body_offset=len(prefix))


def _effective_threshold(n_sq: int, base: float) -> float:
    if n_sq >= LONG_SPAN_MIN_SQ:
        return min(base, LONG_FUZZY_THRESHOLD)
    return base


def _match_squashed(span_plain: str, page: _Page) -> SpanAlignment | None:
    """Steps 1-2: exact whitespace-insensitive match, then dehyphenated."""
    span_sq, _ = _squash(span_plain)
    if not span_sq:
        return None
    hits = _find_all(span_sq, page.sq)
    if hits:
        status = "exact" if len(hits) == 1 else "multiple"
        n_matches = len(hits) if len(hits) > 1 else None
        h = _pick_hit(hits, len(span_sq), page.sq_map, page.body_offset)
        return _clip_to_body(
            page.sq_map, h, h + len(span_sq), page.body_offset,
            status=status, n_matches=n_matches,
        )
    span_dh, _ = _squash(span_plain, drop_hyphens=True)
    if span_dh:
        hits = _find_all(span_dh, page.dh)
        if hits:
            status = "dehyphen" if len(hits) == 1 else "multiple"
            n_matches = len(hits) if len(hits) > 1 else None
            h = _pick_hit(hits, len(span_dh), page.dh_map, page.body_offset)
            return _clip_to_body(
                page.dh_map, h, h + len(span_dh), page.body_offset,
                status=status, n_matches=n_matches,
            )
    return None


def _match_fuzzy(span_plain: str, page: _Page, base_threshold: float) -> SpanAlignment | None:
    """Step 4: anchored SequenceMatcher window over the squashed,
    dehyphenated page, with the length-dependent acceptance threshold."""
    span_sq, _ = _squash(span_plain)
    span_dh, _ = _squash(span_plain, drop_hyphens=True)
    needle = span_dh or span_sq
    if not needle:
        return None
    found = _fuzzy_locate(needle, page.dh, _effective_threshold(len(needle), base_threshold))
    if found is None:
        return None
    s, e, ratio = found
    return _clip_to_body(
        page.dh_map, s, e, page.body_offset, status="fuzzy", ratio=round(ratio, 4)
    )


def _align_unique(
    span_plain: str, page: _Page, base_threshold: float
) -> SpanAlignment | None:
    """Exact/dehyphen/fuzzy cascade, accepting only unique in-body matches
    (used for prefix-stripped remainders and sentence pieces)."""
    al = _match_squashed(span_plain, page) or _match_fuzzy(span_plain, page, base_threshold)
    if al is None or al.in_title or al.status == "multiple":
        return None
    return al


def _strip_known_prefix(
    span_plain: str,
    span_sq: str,
    span_map: list[int],
    title: str | None,
    author: str | None,
) -> str | None:
    """Step 3 helper: remove a leading title [+ author] prefix from the span.

    The comparison is whitespace-insensitive (on the squashed strings); a
    period separating the prefix from the span body ("... Henri Losch. Wéi
    scho gesot ...") is removed as well. Returns the remaining span text,
    or None if the span does not start with a known prefix.
    """
    candidates: list[str] = []
    if title and author:
        candidates.append(title + " " + author)
    if title:
        candidates.append(title)
    for cand in candidates:
        cand_sq, _ = _squash(cand)
        if not cand_sq or not span_sq.startswith(cand_sq):
            continue
        if len(span_sq) <= len(cand_sq):
            continue  # the span IS the prefix; handled by in_title matching
        rest = span_plain[span_map[len(cand_sq)] :]
        rest = rest.lstrip().lstrip(".").lstrip()
        if rest:
            return rest
    return None


def _sentence_pieces(span_plain: str) -> list[str]:
    """Split a span into sentence-like pieces for the partial fallback.

    Cuts after sentence-final punctuation (optionally followed by a closing
    quote) when the next character is whitespace, an uppercase letter, or an
    opening quote — annotated concatenations frequently omit the space
    ("... eng gutt Nummer.Hien hat bewisen ..."). Spaced dashes and "[…]"
    omission markers are removed as separators in their own right.
    """
    cuts = {0, len(span_plain)}
    for m in _SENT_END_RE.finditer(span_plain):
        e = m.end()
        if e < len(span_plain):
            nxt = span_plain[e]
            if nxt.isspace() or nxt.isupper() or nxt in _OPENING_QUOTES:
                cuts.add(e)
    for m in _SEPARATOR_RE.finditer(span_plain):
        cuts.add(m.start())
        cuts.add(m.end())
    pos = sorted(cuts)
    pieces = [span_plain[a:b].strip() for a, b in zip(pos, pos[1:])]
    return [p for p in pieces if p]


def _align_partial(
    span_plain: str, page: _Page, base_threshold: float
) -> SpanAlignment | None:
    """Step 5: align the longest sentence piece of a concatenated span.

    Pieces of >= PARTIAL_MIN_SQ squashed chars are tried longest-first; the
    first unique alignment is returned with status "fuzzy" and partial=True
    (its offsets cover only that piece). Requires at least two pieces —
    a single-sentence span is not a concatenation, just a failed match.
    """
    pieces = _sentence_pieces(span_plain)
    if len(pieces) < 2:
        return None
    sized = [(p, len(_squash(p)[0])) for p in pieces]
    candidates = [(p, n) for p, n in sized if n >= PARTIAL_MIN_SQ]
    candidates.sort(key=lambda t: t[1], reverse=True)
    for piece, _n in candidates:
        al = _align_unique(piece, page, base_threshold)
        if al is not None:
            return SpanAlignment(
                start=al.start, end=al.end, status="fuzzy",
                ratio=al.ratio, partial=True,
            )
    return None


def locate_span(
    span_raw: str,
    body: str,
    title: str | None = None,
    author: str | None = None,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
) -> SpanAlignment:
    """Locate an annotated span inside the clean text *body*.

    *span_raw* is the raw annotated substring (tags included); *body*,
    *title* and *author* come from ``load_clean_text``. See module
    docstring for the cascade of strategies and statuses.
    """
    span_plain = strip_tags(span_raw)
    span_sq, span_map = _squash(span_plain)
    if not span_sq:
        return SpanAlignment(start=None, end=None, status="empty")

    page = _build_page(body, title, author)

    # 1.-2. Exact whitespace-insensitive match, then dehyphenated.
    al = _match_squashed(span_plain, page)
    if al is not None:
        return al

    # 3. Strip a leading title [+ author] prefix from the span and align
    #    the remainder (inner status kept: exact stays exact).
    rest = _strip_known_prefix(span_plain, span_sq, span_map, title, author)
    if rest is not None:
        al = _align_unique(rest, page, fuzzy_threshold)
        if al is not None:
            return al

    # 4. Fuzzy: anchored SequenceMatcher window over the squashed,
    #    dehyphenated page (length-dependent threshold).
    al = _match_fuzzy(span_plain, page, fuzzy_threshold)
    if al is not None:
        return al

    # 5. Partial: concatenated spans — align the longest sentence piece.
    al = _align_partial(span_plain, page, fuzzy_threshold)
    if al is not None:
        return al

    return SpanAlignment(start=None, end=None, status="unresolved")
