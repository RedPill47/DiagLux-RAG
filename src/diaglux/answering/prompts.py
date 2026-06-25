"""Prompt construction for the DiagLux answering harness.

The prompt is FIXED by docs/CONTRACTS.md (section "Prompt (fixed, from the
concept doc Section 7)"). Do not edit the templates without updating the
contract; the template hash is stored in every run's config sidecar so any
drift is detectable.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Optional

# Open-book template, verbatim from docs/CONTRACTS.md.
PROMPT_TEMPLATE = (
    "You are answering a Luxembourgish reading-comprehension question.\n"
    "Use only the provided context.\n"
    "Choose exactly one answer: A, B, C, or D.\n"
    "Return only the letter of the correct answer.\n"
    "\n"
    "Context: {context}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Options:\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "C. {option_c}\n"
    "D. {option_d}\n"
    "\n"
    "Answer:"
)

# Closed-book variant: omits the Context block and the
# "Use only the provided context." line (per docs/CONTRACTS.md).
CLOSED_BOOK_TEMPLATE = (
    "You are answering a Luxembourgish reading-comprehension question.\n"
    "Choose exactly one answer: A, B, C, or D.\n"
    "Return only the letter of the correct answer.\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Options:\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "C. {option_c}\n"
    "D. {option_d}\n"
    "\n"
    "Answer:"
)


def build_prompt(question_record: Mapping, context: Optional[str] = None) -> str:
    """Build the exact contract prompt for one question record.

    ``question_record`` is one line of ``outputs/processed/questions.jsonl``
    (must contain ``question`` and ``presented`` with keys A-D). The options
    used are the PRESENTED (shuffled) ones, never the stored semantic order.

    ``context=None`` produces the closed-book variant. An empty string is a
    valid (empty) open-book context and keeps the Context block.
    """
    presented = question_record["presented"]
    fields = {
        "question": question_record["question"],
        "option_a": presented["A"],
        "option_b": presented["B"],
        "option_c": presented["C"],
        "option_d": presented["D"],
    }
    if context is None:
        return CLOSED_BOOK_TEMPLATE.format(**fields)
    return PROMPT_TEMPLATE.format(context=context, **fields)


def prompt_template_hash() -> str:
    """Stable short hash of both template variants, stored in run sidecars."""
    blob = (PROMPT_TEMPLATE + "\n===\n" + CLOSED_BOOK_TEMPLATE).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
