"""Tokenizer behaviour: apostrophes, diacritics, casing, NFC, analyzers."""

import pytest

from diaglux.retrieval.tokenize import (
    char_ngram_tokenize,
    get_tokenizer,
    normalize,
    word_tokenize,
)


def test_clitic_apostrophe_split_ascii():
    assert word_tokenize("d'Schoul") == ["d'", "schoul"]


def test_clitic_apostrophe_split_typographic():
    # U+2019 must behave exactly like ASCII apostrophe
    assert word_tokenize("d’Schoul") == ["d'", "schoul"]


def test_other_clitics():
    assert word_tokenize("z'iessen an t'ass") == ["z'", "iessen", "an", "t'", "ass"]


def test_trailing_apostrophe_is_punctuation():
    # closing quote is not a clitic: apostrophe not followed by a letter
    assert word_tokenize("'Moien' sot hatt") == ["moien", "sot", "hatt"]


def test_diacritics_are_kept():
    assert word_tokenize("säin Frënd ass glécklech") == ["säin", "frënd", "ass", "glécklech"]


def test_lowercasing():
    assert word_tokenize("Wat ass DAT") == ["wat", "ass", "dat"]


def test_nfc_normalization():
    # decomposed e + COMBINING ACUTE must equal precomposed é
    assert word_tokenize("glécklech") == word_tokenize("glécklech")


def test_punctuation_dropped_digits_kept():
    assert word_tokenize("Hien ass 12 Joer al!") == ["hien", "ass", "12", "joer", "al"]


def test_no_stopwords_by_default():
    assert "ass" in word_tokenize("d'Schoul ass grouss")


def test_pluggable_stopwords():
    tokens = word_tokenize("d'Schoul ass grouss", stopwords={"d'", "ass"})
    assert tokens == ["schoul", "grouss"]


def test_normalize_maps_typographic_apostrophe():
    assert normalize("D’Kand") == "d'kand"


def test_char_ngrams_counts_and_content():
    grams = char_ngram_tokenize("Schoul", n_min=3, n_max=5)
    # len 6 word: 4 trigrams + 3 four-grams + 2 five-grams = 9
    assert len(grams) == 9
    assert "sch" in grams and "oul" in grams and "schou" in grams
    assert all(3 <= len(g) <= 5 for g in grams)


def test_char_ngrams_short_words_emitted_whole():
    # "d'" (len 2) survives as itself; clitic split still applies first
    grams = char_ngram_tokenize("d'Schoul an", n_min=3, n_max=5)
    assert "d'" in grams and "an" in grams


def test_char_ngrams_invalid_range():
    with pytest.raises(ValueError):
        char_ngram_tokenize("Schoul", n_min=4, n_max=3)


def test_get_tokenizer_dispatch():
    assert get_tokenizer("word")("d'Schoul") == ["d'", "schoul"]
    assert "sch" in get_tokenizer("char_ngram")("Schoul")
    with pytest.raises(ValueError):
        get_tokenizer("nope")
