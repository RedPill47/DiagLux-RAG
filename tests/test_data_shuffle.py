"""Option shuffling: determinism, distribution, and round-trip."""

from diaglux.data.shuffle import (
    SEMANTIC_TYPES,
    letter_to_semantic,
    shuffle_options,
)

OPTIONS = {
    "correct": "the right one",
    "misunderstand": "the misreading",
    "distractor_span": "the lure",
    "no_support": "the fabrication",
}


def test_round_trip_semantic_letter_semantic():
    for i in range(50):
        qid = f"text{i % 16 + 1}_q{i:02d}"
        presented, permutation, gold = shuffle_options(qid, OPTIONS)
        assert sorted(permutation) == sorted(SEMANTIC_TYPES)
        for idx, letter in enumerate("ABCD"):
            sem = letter_to_semantic(letter, permutation)
            assert permutation[idx] == sem
            assert presented[letter] == OPTIONS[sem]
        assert letter_to_semantic(gold, permutation) == "correct"
        assert presented[gold] == OPTIONS["correct"]


def test_deterministic_per_question():
    a = shuffle_options("text3_q07", OPTIONS)
    b = shuffle_options("text3_q07", OPTIONS)
    assert a == b


def test_global_seed_changes_permutation_somewhere():
    perms_13 = [shuffle_options(f"text1_q{i:02d}", OPTIONS, 13)[1] for i in range(40)]
    perms_99 = [shuffle_options(f"text1_q{i:02d}", OPTIONS, 99)[1] for i in range(40)]
    assert perms_13 != perms_99


def test_correct_not_always_first():
    golds = {shuffle_options(f"text2_q{i:02d}", OPTIONS)[2] for i in range(40)}
    assert len(golds) > 1  # the positional answer key must be broken up
