"""KB line parsing and tag extraction on fixture data."""

import pytest

from diaglux.data.kb import parse_kb_line
from diaglux.data.tags import extract_tags, strip_tags, tag_categories

FIXTURE_LINE = (
    "'Eng Geschicht': ["
    "{'textTitle': 'Eng Geschicht', 'cognitiveType': 'Retrieve', "
    "'criticalSpan': 'Danzen ass <LEX-FALSE-FRIEND> souwisou mäin Hobby "
    "<LEX-LOAN-DE> , a <ORTHO-PHONO-DIVERGE> <MORPH-N-RULE> esou.', "
    "'distractorSpan': 'En aneren <SYN-VERB-SEP> Saz.', "
    "'question': 'Waat ass säin Hobby?', "
    "'A_Correct': 'Danzen.', 'B_Misunderstand': 'Turnen.', "
    "'C_Distractor_Span': 'Reesen.', 'D_No_Support': 'Liesen.'}]"
)


def test_parse_kb_line_fixture():
    title, questions = parse_kb_line(FIXTURE_LINE)
    assert title == "Eng Geschicht"
    assert len(questions) == 1
    q = questions[0]
    assert q["cognitiveType"] == "Retrieve"
    assert q["A_Correct"] == "Danzen."
    assert "<LEX-FALSE-FRIEND>" in q["criticalSpan"]


def test_parse_kb_line_rejects_bad_schema():
    bad = "'T': [{'textTitle': 'T', 'cognitiveType': 'Retrieve'}]"
    with pytest.raises(ValueError, match="schema violation"):
        parse_kb_line(bad)


def test_parse_kb_line_rejects_bad_cognitive_type():
    line = FIXTURE_LINE.replace("'Retrieve'", "'Guess'")
    with pytest.raises(ValueError, match="cognitiveType"):
        parse_kb_line(line)


def test_strip_and_extract_tags():
    span = "Danzen ass <LEX-FALSE-FRIEND> souwisou <LEX-LOAN-DE> , a <MORPH-N-RULE> <LEX-FALSE-FRIEND> esou."
    assert "<" not in strip_tags(span)
    assert strip_tags(span).split() == ["Danzen", "ass", "souwisou", ",", "a", "esou."]
    tags = extract_tags(span)
    assert tags == ["LEX-FALSE-FRIEND", "LEX-LOAN-DE", "MORPH-N-RULE"]  # deduped, ordered
    assert tag_categories(tags) == ["LEX", "MORPH"]


def test_tag_categories_canonical_order():
    assert tag_categories(["ORTHO-PHONO-DIVERGE", "LEX-LOAN-DE", "DISC-COREF-AMBIG"]) == [
        "LEX",
        "ORTHO",
        "DISC",
    ]
