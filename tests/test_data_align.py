"""Span alignment on a synthetic clean text."""

from diaglux.data.align import locate_span

TITLE = "Eng kleng Geschicht"
AUTHOR = "Mim Keseler"
BODY = (
    "De Schoulmeeschter huet eis gëschter eng laang Geschicht erzielt.\n"
    "Ech sinn esou frou, dass ech\n"
    "hien kennen. D'Telefons-\n"
    "buch louch um Dësch, an et war eng laang Nuecht.\n"
    "Ech sinn nees frou. Ech sinn nees frou.\n"
    "Hatt wuar ganz eleng doheem an der grousser Stuff.\n"
)


def _check(alignment, body=BODY):
    """Located offsets must point at real body content."""
    assert alignment.start is not None and alignment.end is not None
    assert 0 <= alignment.start < alignment.end <= len(body)


def test_exact_across_line_break_with_tags():
    span = "esou frou, dass <LEX-FALSE-FRIEND> ech hien kennen"
    al = locate_span(span, BODY, title=TITLE)
    assert al.status == "exact"
    _check(al)
    assert " ".join(BODY[al.start : al.end].split()) == "esou frou, dass ech hien kennen"


def test_dehyphenated_match():
    # The annotated span lost the line-break hyphen ("Telefonsbuch").
    span = "D'Telefonsbuch louch um Dësch"
    al = locate_span(span, BODY, title=TITLE)
    assert al.status == "dehyphen"
    _check(al)
    assert BODY[al.start : al.end].startswith("D'Telefons-")
    assert BODY[al.start : al.end].endswith("Dësch")


def test_multiple_occurrences():
    al = locate_span("Ech sinn nees frou.", BODY, title=TITLE)
    assert al.status == "multiple"
    assert al.n_matches == 2
    _check(al)
    assert BODY[al.start : al.end] == "Ech sinn nees frou."


def test_fuzzy_spelling_divergence():
    # Annotated "war" vs clean "wuar" (and an extra word changed).
    span = "Hatt war ganz eleng doheem an der grousser Stuff."
    al = locate_span(span, BODY, title=TITLE)
    assert al.status == "fuzzy"
    assert al.ratio is not None and al.ratio >= 0.85
    _check(al)
    assert "eleng doheem" in BODY[al.start : al.end]


def test_span_is_title():
    al = locate_span("Eng kleng <ORTHO-PHONO-DIVERGE> Geschicht", BODY, title=TITLE)
    assert al.status == "exact"
    assert al.in_title is True
    assert al.start is None and al.end is None


def test_empty_span():
    al = locate_span("  <LEX-LOAN-DE>  ", BODY, title=TITLE)
    assert al.status == "empty"
    assert al.start is None and al.end is None


def test_unresolved():
    al = locate_span("Dëse Saz steet néierens am Text a passt och net.", BODY, title=TITLE)
    assert al.status == "unresolved"
    assert al.start is None and al.end is None
    assert al.partial is False


# --- title/author prefix handling ----------------------------------------


def test_prefix_title_author_clipped_to_body():
    # Annotated span starts at the very top of the annotated file:
    # title + author + first body sentence. The match page contains the
    # prefix, so the span matches exactly and is clipped to the body.
    span = (
        "Eng kleng Geschicht Mim Keseler "
        "Ech sinn esou frou, dass ech hien kennen."
    )
    al = locate_span(span, BODY, title=TITLE, author=AUTHOR)
    assert al.status == "exact"
    assert al.partial is False
    _check(al)
    assert " ".join(BODY[al.start : al.end].split()) == (
        "Ech sinn esou frou, dass ech hien kennen."
    )


def test_prefix_strip_with_separating_period():
    # A period between the author name and the body content breaks the
    # whole-page match; the title+author prefix is stripped from the SPAN
    # and the remainder still aligns exactly (status semantics kept).
    span = (
        "Eng kleng Geschicht Mim Keseler. "
        "Ech sinn esou frou, dass ech hien kennen."
    )
    al = locate_span(span, BODY, title=TITLE, author=AUTHOR)
    assert al.status == "exact"
    assert al.partial is False
    _check(al)
    assert " ".join(BODY[al.start : al.end].split()) == (
        "Ech sinn esou frou, dass ech hien kennen."
    )


def test_prefix_strip_title_only_keeps_inner_status():
    # Title-prefixed span whose remainder needs the dehyphen step: the
    # inner cascade decides the status, not the prefix handling.
    span = "Eng kleng Geschicht D'Telefonsbuch louch um Dësch"
    al = locate_span(span, BODY, title=TITLE)
    assert al.status == "dehyphen"
    _check(al)
    assert BODY[al.start : al.end].startswith("D'Telefons-")


# --- partial alignment of concatenated spans ------------------------------


def test_partial_concatenated_span():
    # The annotated span glues together two non-contiguous body sentences;
    # the longest sentence piece is aligned instead, with partial=True.
    span = (
        "De Schoulmeeschter huet eis gëschter eng laang Geschicht erzielt. "
        "Hatt wuar ganz eleng doheem an der grousser Stuff."
    )
    al = locate_span(span, BODY, title=TITLE, author=AUTHOR)
    assert al.status == "fuzzy"
    assert al.partial is True
    _check(al)
    # The longest piece (the first sentence) was aligned, exactly.
    assert BODY[al.start : al.end] == (
        "De Schoulmeeschter huet eis gëschter eng laang Geschicht erzielt."
    )


def test_partial_needs_substantial_pieces():
    # All pieces are below the 30-squashed-char floor: no partial rescue.
    al = locate_span("Net hei. Och net do. Guer net.", BODY, title=TITLE, author=AUTHOR)
    assert al.status == "unresolved"
    assert al.partial is False


# --- length-dependent fuzzy threshold -------------------------------------

# A real body stretch (>= 60 squashed chars) with scattered single-character
# divergences, tuned to a similarity of ~0.83: below the strict 0.85
# threshold but above the long-span 0.80 threshold.
_LONG_DIVERGENT = (
    "qn et wqr eng qaang quechq. Ech sqnn neqs froq. Ech sqnn neqs froq. "
    "Hatt quar gqnz elqng doqeem aq der grousser Stuff."
)
# The same mutation rate on a short (< 60 squashed chars) span.
_SHORT_DIVERGENT = "qatt wqar gaqz eleqg dohqem an qer grqusseq Stuff."


def test_long_span_accepts_lowered_threshold():
    al = locate_span(_LONG_DIVERGENT, BODY, title=TITLE, author=AUTHOR)
    assert al.status == "fuzzy"
    assert al.partial is False
    assert al.ratio is not None and 0.80 <= al.ratio < 0.85
    _check(al)


def test_short_span_keeps_strict_threshold():
    # Equally divergent but short: must NOT be accepted at 0.80.
    al = locate_span(_SHORT_DIVERGENT, BODY, title=TITLE, author=AUTHOR)
    assert al.status == "unresolved"
    assert al.start is None and al.end is None
