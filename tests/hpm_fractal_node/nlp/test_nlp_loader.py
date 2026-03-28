"""Tests for nlp_loader: vocab, encoding, sentence generation."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from hpm_fractal_node.nlp.nlp_loader import (
    VOCAB, VOCAB_INDEX, D,
    encode_context_window,
    generate_sentences,
    category_names,
)


def test_vocab_size():
    assert len(VOCAB) == 107


def test_d_is_428():
    assert D == 428


def test_vocab_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB))


def test_special_tokens_present():
    for tok in ("<start>", "<end>", "<unknown>"):
        assert tok in VOCAB_INDEX


def test_encode_context_window_shape():
    vec = encode_context_window("the", "barked", left2="<start>", right2="at")
    assert vec.shape == (D,)
    assert vec.dtype == np.float32


def test_encode_context_window_one_hot():
    vec = encode_context_window("the", "barked", left2="<start>", right2="at")
    # Each of 4 slots should have exactly one 1 (known words)
    for i in range(4):
        slot = vec[i * 107: (i + 1) * 107]
        assert slot.sum() == 1.0


def test_encode_unknown_word():
    vec = encode_context_window("zzz_unknown", "barked")
    # left_1 is slot 1 (indices 107..213), should encode as <unknown>
    unk_idx = VOCAB_INDEX["<unknown>"]
    assert vec[107 + unk_idx] == 1.0


def test_generate_sentences_returns_list():
    sentences = generate_sentences(seed=42)
    assert len(sentences) >= 2000  # must reach N_SAMPLES


def test_generate_sentences_has_labels():
    sentences = generate_sentences(seed=42)
    obs, true_word, category = sentences[0]
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (D,)
    assert isinstance(true_word, str)
    assert category in category_names()


def test_generate_sentences_reproducible():
    s1 = generate_sentences(seed=0)
    s2 = generate_sentences(seed=0)
    assert len(s1) == len(s2)
    np.testing.assert_array_equal(s1[0][0], s2[0][0])


def test_category_names_returns_7():
    cats = category_names()
    assert len(cats) == 7
    expected = {"animal", "adult", "child_person", "family", "food", "object", "place"}
    assert set(cats) == expected
