"""Seeded per-question option shuffling.

The stored option order is positionally informative (A is always correct),
so options MUST be shuffled before presentation. Each question gets its own
deterministic permutation derived from a stable hash of its question_id
mixed with the global seed (13), so a single question's presentation never
depends on processing order or on other questions.
"""

from __future__ import annotations

import hashlib
import random

GLOBAL_SEED = 13
SEMANTIC_TYPES = ("correct", "misunderstand", "distractor_span", "no_support")
LETTERS = ("A", "B", "C", "D")


def question_seed(question_id: str, global_seed: int = GLOBAL_SEED) -> int:
    """Stable 64-bit seed from question_id + global seed (hash() is
    process-randomized, so sha256 is used instead)."""
    digest = hashlib.sha256(f"{question_id}|{global_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def shuffle_options(
    question_id: str,
    options: dict[str, str],
    global_seed: int = GLOBAL_SEED,
) -> tuple[dict[str, str], list[str], str]:
    """Shuffle one question's options.

    Returns ``(presented, permutation, gold_letter)`` where
    ``presented[letter]`` is the option text shown as that letter,
    ``permutation[i]`` is the semantic type shown as letter ``chr(65 + i)``,
    and ``gold_letter`` is the letter carrying the "correct" option.
    """
    missing = [t for t in SEMANTIC_TYPES if t not in options]
    if missing:
        raise ValueError(f"options missing semantic types: {missing}")
    permutation = list(SEMANTIC_TYPES)
    random.Random(question_seed(question_id, global_seed)).shuffle(permutation)
    presented = {LETTERS[i]: options[sem] for i, sem in enumerate(permutation)}
    gold_letter = LETTERS[permutation.index("correct")]
    return presented, permutation, gold_letter


def letter_to_semantic(letter: str, permutation: list[str]) -> str:
    """Map a presented letter ("A"-"D") back to its semantic option type."""
    idx = ord(letter.upper()) - 65
    if not 0 <= idx < len(permutation):
        raise ValueError(f"invalid option letter: {letter!r}")
    return permutation[idx]
