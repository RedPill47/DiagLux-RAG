"""DiagLux-RAG data pipeline: parsers, span alignment, shuffling, chunking.

Owns ``src/diaglux/data/`` and ``outputs/processed/`` (see docs/CONTRACTS.md).
"""

from diaglux.data.texts import find_data_root, load_clean_text
from diaglux.data.kb import load_questions, parse_kb_line
from diaglux.data.tags import strip_tags, extract_tags, tag_categories
from diaglux.data.align import locate_span, SpanAlignment
from diaglux.data.shuffle import (
    GLOBAL_SEED,
    SEMANTIC_TYPES,
    shuffle_options,
    letter_to_semantic,
)
from diaglux.data.chunking import (
    Chunk,
    chunk_paragraph,
    chunk_overlap,
    chunk_sentence,
    check_full_coverage,
)

__all__ = [
    "find_data_root",
    "load_clean_text",
    "load_questions",
    "parse_kb_line",
    "strip_tags",
    "extract_tags",
    "tag_categories",
    "locate_span",
    "SpanAlignment",
    "GLOBAL_SEED",
    "SEMANTIC_TYPES",
    "shuffle_options",
    "letter_to_semantic",
    "Chunk",
    "chunk_paragraph",
    "chunk_overlap",
    "chunk_sentence",
    "check_full_coverage",
]
