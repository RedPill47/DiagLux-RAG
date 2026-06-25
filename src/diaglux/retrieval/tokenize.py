"""Luxembourgish-aware text preprocessing for retrieval.

There is no standard Luxembourgish analyzer (review_and_plan Section 2.9), so
every preprocessing decision is made explicit here:

1. **Unicode NFC normalization.** The clean text bodies are NFC-normalized
   (docs/CONTRACTS.md), so queries must be normalized identically or composed
   vs. decomposed diacritics ("e" + U+0301 vs. "é") would fail to match.

2. **Lowercasing.** Applied unconditionally. Luxembourgish nouns are
   capitalized (as in German), but the dataset's questions/options use
   informal, inconsistent casing, and ``text13`` ("Poker") is all-lowercase
   free verse — case is therefore not a reliable signal and lowercasing
   maximizes query-document matching. (For text13 this is a no-op.)

3. **Diacritics are KEPT.** Luxembourgish diacritics are phonemic and
   contrastive (``säin`` "his" vs. hypothetical ``sain``; ``é``/``ë``/``ä``
   distinguish lexemes). Stripping them would conflate distinct words. The
   query-document orthography gap (e.g. informal ``sein`` vs. literary
   ``säin``) is real, but we treat it as a *finding* about exact-match
   retrieval rather than papering over it; the char n-gram analyzer below is
   the documented ablation that partially bridges it.

4. **Clitic apostrophe splitting.** Luxembourgish writes article/pronoun
   clitics with an apostrophe attached to the next word: ``d'Schoul``,
   ``d'Kanner``, ``t'ass``, ``z'iessen``. We split these into the clitic
   (with its apostrophe, e.g. ``d'``) plus the host word (``schoul``), so the
   content word is indexable on its own. Both the ASCII apostrophe ``'``
   (U+0027) and the typographic apostrophe ``’`` (U+2019) are handled; the
   typographic form is mapped to ASCII so ``d'`` from either source is the
   same token. A trailing apostrophe NOT followed by a letter (e.g. a closing
   quote) is treated as punctuation and dropped.

5. **Regex word tokenization.** Tokens are maximal runs of Unicode letters,
   or runs of digits; everything else (punctuation, symbols) is a separator.

6. **No stopword list by default.** There is no standard Luxembourgish
   stopword list; rather than invent one silently, stopword removal is a
   pluggable parameter (``stopwords=``) so a custom list can be ablated.

7. **Character n-gram analyzer (ablation).** ``char_ngram_tokenize`` emits
   character 3–5-grams *within* word tokens (no cross-word n-grams, no
   boundary padding; words shorter than ``n_min`` are emitted whole). This is
   the subword BM25 variant suggested for Luxembourgish's rich compounding
   and orthographic variation (Section 2.9).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Collection, Iterable, List, Optional

# Token = letters followed by a clitic apostrophe that is itself followed by a
# letter (the clitic, apostrophe kept) | run of letters | run of digits.
# [^\W\d_] == "Unicode letter" for the re module.
_TOKEN_RE = re.compile(r"[^\W\d_]+'(?=[^\W\d_])|[^\W\d_]+|\d+")

#: Signature shared by all analyzers: text -> list of token strings.
TokenizerFn = Callable[[str], List[str]]


def normalize(text: str) -> str:
    """NFC-normalize, map typographic apostrophe U+2019 to ASCII, lowercase."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'")
    return text.lower()


def word_tokenize(text: str, stopwords: Optional[Collection[str]] = None) -> List[str]:
    """Tokenize ``text`` into lowercase word tokens (see module docstring).

    Parameters
    ----------
    text : raw text (any casing / apostrophe variant / Unicode form).
    stopwords : optional collection of tokens to drop *after* tokenization
        (compare lowercase, NFC, ASCII-apostrophe forms, e.g. ``{"d'", "an"}``).
        Default ``None`` = no stopword removal (no standard lb list exists).
    """
    tokens = _TOKEN_RE.findall(normalize(text))
    if stopwords:
        stop = set(stopwords)
        tokens = [t for t in tokens if t not in stop]
    return tokens


def char_ngram_tokenize(
    text: str,
    n_min: int = 3,
    n_max: int = 5,
    stopwords: Optional[Collection[str]] = None,
) -> List[str]:
    """Character n-gram analyzer (ablation alternative to word tokens).

    Word-tokenizes first (so clitic splitting / lowercasing / NFC apply),
    optionally removes stopwords at the *word* level, then emits all character
    n-grams with ``n_min <= n <= n_max`` inside each word token. Words shorter
    than ``n_min`` are emitted whole so they are not lost. The clitic
    apostrophe is part of its token (``d'`` -> emitted whole, len 2 < 3).
    """
    if n_min < 1 or n_max < n_min:
        raise ValueError(f"invalid n-gram range [{n_min}, {n_max}]")
    grams: List[str] = []
    for word in word_tokenize(text, stopwords=stopwords):
        if len(word) < n_min:
            grams.append(word)
            continue
        for n in range(n_min, min(n_max, len(word)) + 1):
            for i in range(len(word) - n + 1):
                grams.append(word[i : i + n])
    return grams


def get_tokenizer(
    analyzer: str = "word",
    stopwords: Optional[Collection[str]] = None,
    n_min: int = 3,
    n_max: int = 5,
) -> TokenizerFn:
    """Return a ``text -> tokens`` callable for ``analyzer`` in {word, char_ngram}."""
    if analyzer == "word":
        return lambda text: word_tokenize(text, stopwords=stopwords)
    if analyzer == "char_ngram":
        return lambda text: char_ngram_tokenize(
            text, n_min=n_min, n_max=n_max, stopwords=stopwords
        )
    raise ValueError(f"unknown analyzer {analyzer!r} (expected 'word' or 'char_ngram')")


def tokenize_corpus(
    texts: Iterable[str], tokenizer: Optional[TokenizerFn] = None
) -> List[List[str]]:
    """Tokenize an iterable of documents with ``tokenizer`` (default: word)."""
    tok = tokenizer or word_tokenize
    return [tok(t) for t in texts]
