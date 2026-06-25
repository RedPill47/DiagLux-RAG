"""Prompt construction must match docs/CONTRACTS.md verbatim.

The expected strings below are written out literally from the contract's
"Prompt (fixed, from the concept doc Section 7)" section -- they are NOT
derived from the module's own template constants, so any drift in
prompts.py is caught.
"""

from answering_testutils import make_questions
from diaglux.answering.prompts import build_prompt, prompt_template_hash

QUESTION = {
    "question": "Wat huet d'Catherine fonnt?",
    "presented": {
        "A": "E Bridfchen.",
        "B": "E Schlessel.",
        "C": "Eng Kaz.",
        "D": "Naischt.",
    },
}

# Verbatim from docs/CONTRACTS.md, with the placeholders filled in.
EXPECTED_OPEN_BOOK = (
    "You are answering a Luxembourgish reading-comprehension question.\n"
    "Use only the provided context.\n"
    "Choose exactly one answer: A, B, C, or D.\n"
    "Return only the letter of the correct answer.\n"
    "\n"
    "Context: Si souz am Gaart an huet e Bridfchen fonnt.\n"
    "\n"
    "Question: Wat huet d'Catherine fonnt?\n"
    "\n"
    "Options:\n"
    "A. E Bridfchen.\n"
    "B. E Schlessel.\n"
    "C. Eng Kaz.\n"
    "D. Naischt.\n"
    "\n"
    "Answer:"
)

# Closed-book variant: omits the Context block AND the
# "Use only the provided context." line.
EXPECTED_CLOSED_BOOK = (
    "You are answering a Luxembourgish reading-comprehension question.\n"
    "Choose exactly one answer: A, B, C, or D.\n"
    "Return only the letter of the correct answer.\n"
    "\n"
    "Question: Wat huet d'Catherine fonnt?\n"
    "\n"
    "Options:\n"
    "A. E Bridfchen.\n"
    "B. E Schlessel.\n"
    "C. Eng Kaz.\n"
    "D. Naischt.\n"
    "\n"
    "Answer:"
)


def test_open_book_prompt_matches_contract_verbatim():
    prompt = build_prompt(
        QUESTION, context="Si souz am Gaart an huet e Bridfchen fonnt."
    )
    assert prompt == EXPECTED_OPEN_BOOK


def test_closed_book_prompt_matches_contract_verbatim():
    prompt = build_prompt(QUESTION, context=None)
    assert prompt == EXPECTED_CLOSED_BOOK


def test_closed_book_omits_context_block_and_use_only_line():
    prompt = build_prompt(QUESTION, context=None)
    assert "Context:" not in prompt
    assert "Use only the provided context." not in prompt


def test_empty_string_context_keeps_context_block():
    # Empty string is a valid (empty) open-book context, distinct from None.
    prompt = build_prompt(QUESTION, context="")
    assert "Context: \n" in prompt
    assert "Use only the provided context." in prompt


def test_prompt_uses_presented_options_not_semantic_order():
    question = make_questions()[0]  # permutation puts "correct" at B
    prompt = build_prompt(question, context="ctx")
    for letter in "ABCD":
        assert f"{letter}. {question['presented'][letter]}" in prompt
    # The semantic-order dict must not leak in as A-D order.
    assert f"A. {question['options']['correct']}" not in prompt


def test_prompt_template_hash_is_stable_short_hex():
    h = prompt_template_hash()
    assert h == prompt_template_hash()
    assert len(h) == 16
    int(h, 16)  # raises if not hex
