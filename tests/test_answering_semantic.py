"""Letter <-> semantic option type mapping via the stored permutation."""

import pytest

from answering_testutils import SEMANTIC_TYPES, make_questions
from diaglux.answering.runner import _semantic_choice


@pytest.fixture(scope="module")
def questions():
    return make_questions()


def test_gold_letter_maps_to_correct(questions):
    for question in questions:
        assert _semantic_choice(question, question["gold_letter"]) == "correct"


def test_letter_to_semantic_follows_permutation(questions):
    question = questions[0]
    # permutation[i] = semantic type shown as letter chr(65+i)
    assert question["permutation"] == [
        "distractor_span", "correct", "no_support", "misunderstand",
    ]
    assert _semantic_choice(question, "A") == "distractor_span"
    assert _semantic_choice(question, "B") == "correct"
    assert _semantic_choice(question, "C") == "no_support"
    assert _semantic_choice(question, "D") == "misunderstand"


def test_round_trip_semantic_to_letter_and_back(questions):
    for question in questions:
        for semantic in SEMANTIC_TYPES:
            letter = chr(65 + question["permutation"].index(semantic))
            assert _semantic_choice(question, letter) == semantic
        # And every letter maps back to a unique semantic type.
        mapped = [_semantic_choice(question, ltr) for ltr in "ABCD"]
        assert sorted(mapped) == sorted(SEMANTIC_TYPES)


def test_none_letter_maps_to_none(questions):
    assert _semantic_choice(questions[0], None) is None


def test_presented_text_consistent_with_permutation(questions):
    for question in questions:
        for i, semantic in enumerate(question["permutation"]):
            letter = chr(65 + i)
            assert question["presented"][letter] == question["options"][semantic]
