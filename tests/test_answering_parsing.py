"""parse_letter edge cases, including Luxembourgish false positives.

Contract: returns (letter, parse_status) with letter in A-D or None and
parse_status in exact | extracted | unparseable.
"""

import pytest

from diaglux.answering.parsing import parse_letter


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("B", ("B", "exact")),
        ("b", ("B", "exact")),
        ("b.", ("B", "exact")),
        ("D.", ("D", "exact")),
        ("C)", ("C", "exact")),
        ("  A  ", ("A", "exact")),
        ("A:", ("A", "exact")),
    ],
)
def test_exact_single_letter(raw, expected):
    assert parse_letter(raw) == expected


@pytest.mark.parametrize(
    "raw, letter",
    [
        ("Answer: C", "C"),
        ("answer: c", "C"),
        ("The answer is B", "B"),
        ("the answer is b.", "B"),
        ("The correct answer is **B**.", "B"),
        ("**D**", "D"),
        ("**D.**", "D"),
        ("(A)", "A"),
        ("( a )", "A"),
        ("Ech mengen et ass B.", "B"),  # standalone capital token
        ("C. Si geet heem.", "C"),  # option echo
    ],
)
def test_extracted_variants(raw, letter):
    assert parse_letter(raw) == (letter, "extracted")


@pytest.mark.parametrize(
    "raw",
    [
        # Capital A-D embedded inside Luxembourgish words must NOT match.
        "Dat ass eng gutt Iddi.",
        "Den Auto war blo.",
        "Catherine war frou.",
        # Apostrophe contraction guard: "D'" is the definite article.
        "D'Kanner spillen am Gaart.",
        "d'Buch louch um Desch.",
        # Plain garbage.
        "Ech weess et net.",
        "12345 !?",
        "",
        "   ",
    ],
)
def test_unparseable(raw):
    assert parse_letter(raw) == (None, "unparseable")


def test_none_input_is_unparseable():
    assert parse_letter(None) == (None, "unparseable")


@pytest.mark.parametrize(
    "raw, letter",
    [
        # Chain-of-thought that discusses early options then concludes: the
        # concluding letter (last) must win, not an option mentioned first.
        ("Option A is about dancing, but the text supports C.", "C"),
        ("(A) is wrong and (B) is a misreading; the answer is (D).", "D"),
        ("I think it's between B and C. Ultimately, the answer is B.", "B"),
        ("The answer is not A, it's B.", "B"),
        ("Looking at the text... therefore the narrator left. C", "C"),
        ("**A** seems plausible at first, but the correct answer is **D**.", "D"),
    ],
)
def test_chain_of_thought_takes_concluding_letter(raw, letter):
    parsed, status = parse_letter(raw)
    assert parsed == letter
    assert status == "extracted"


def test_lowercase_mid_sentence_does_not_match_standalone():
    # Standalone fallback is uppercase-only; a lone lowercase letter inside
    # a sentence (no answer marker) must not be promoted to an answer.
    assert parse_letter("et ass d a net e") == (None, "unparseable")


def test_negated_letter_is_rejected_not_chosen():
    # "Et ass B, net C." = "it's B, not C": C is negated, so B wins even though
    # C appears later. Negation-awareness, not position, drives the choice.
    assert parse_letter("Et ass B, net C.") == ("B", "extracted")
    assert parse_letter("Net A, mä B.") == ("B", "extracted")
