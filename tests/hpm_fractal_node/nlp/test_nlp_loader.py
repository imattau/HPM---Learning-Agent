"""Tests for nlp_loader.py — D=107 compose_context_node."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from hpm_fractal_node.nlp.nlp_loader import (
    VOCAB, VOCAB_INDEX, VOCAB_SIZE, D,
    compose_context_node,
    generate_sentences,
    category_names,
)


def test_vocab_size():
    assert len(VOCAB) == 107


def test_d_equals_vocab_size():
    """D must be 107, not 428."""
    assert D == 107
    assert D == VOCAB_SIZE


def test_vocab_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB))


def test_special_tokens_present():
    for tok in ("<start>", "<end>", "<unknown>"):
        assert tok in VOCAB_INDEX


def test_compose_context_node_shape():
    """compose_context_node returns shape (107,)."""
    out = compose_context_node("the", "dog", "barked", "<end>")
    assert out.shape == (D,)


def test_compose_context_node_dtype():
    """compose_context_node returns float64."""
    out = compose_context_node("the", "dog", "barked", "<end>")
    assert out.dtype == np.float64


def test_compose_context_node_weights_sum_to_one():
    """Slot weights 0.2+0.35+0.35+0.10 sum to 1.0; so all-distinct output sums to 1.0."""
    # Use four distinct words so no index overlaps
    out = compose_context_node("the", "dog", "barked", "park")
    assert abs(out.sum() - 1.0) < 1e-10


def test_compose_context_node_repeated_word_accumulates():
    """Same word in left1 and right1 accumulates 0.35+0.35=0.70 at that index."""
    word = "dog"
    idx = VOCAB_INDEX["dog"]
    out = compose_context_node("<start>", word, word, "<end>")
    # left2=<start>=0.20, left1=dog=0.35, right1=dog=0.35, right2=<end>=0.10
    # dog index should hold 0.70
    assert abs(out[idx] - 0.70) < 1e-10


def test_compose_context_node_known_values():
    """Spot-check exact weight at a specific index."""
    # left2="the" (w=0.20), left1="the" (w=0.35), right1="cat" (w=0.35), right2="<end>" (w=0.10)
    out = compose_context_node("the", "the", "cat", "<end>")
    the_idx = VOCAB_INDEX["the"]
    cat_idx = VOCAB_INDEX["cat"]
    assert abs(out[the_idx] - 0.55) < 1e-10   # 0.20 + 0.35
    assert abs(out[cat_idx] - 0.35) < 1e-10


def test_compose_context_node_unknown_word():
    """Unknown word maps to <unknown> index."""
    unk_idx = VOCAB_INDEX["<unknown>"]
    out = compose_context_node("zzz_unknown", "dog", "barked", "<end>")
    # left2=zzz_unknown gets weight 0.20 at <unknown> index
    assert abs(out[unk_idx] - 0.20) < 1e-10


def test_generate_sentences_returns_2000():
    sentences = generate_sentences(seed=42)
    assert len(sentences) == 2000


def test_generate_sentences_returns_d107_vectors():
    """generate_sentences() must produce D=107 vectors after refactor."""
    data = generate_sentences(seed=42)
    assert len(data) == 2000
    vec, word, cat = data[0]
    assert vec.shape == (D,)
    assert vec.dtype == np.float64


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


def test_no_encode_context_window():
    """encode_context_window must NOT be importable (removed in refactor)."""
    import hpm_fractal_node.nlp.nlp_loader as mod
    assert not hasattr(mod, "encode_context_window"), \
        "encode_context_window should be removed in D=107 refactor"
