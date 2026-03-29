"""Tests for code_loader.py"""

import numpy as np
import pytest
from hpm_fractal_node.code.code_loader import (
    VOCAB, VOCAB_INDEX, VOCAB_SIZE, D,
    compose_context_node, generate_code_snippets,
)


def test_vocab_size():
    assert VOCAB_SIZE == 70
    assert len(VOCAB) == 70


def test_vocab_no_duplicates():
    assert len(set(VOCAB)) == 70


def test_vocab_index_consistent():
    for i, token in enumerate(VOCAB):
        assert VOCAB_INDEX[token] == i


def test_d_equals_vocab_size():
    assert D == VOCAB_SIZE


def test_vocab_contains_keywords():
    for kw in ["if", "for", "while", "def", "class", "return", "import"]:
        assert kw in VOCAB_INDEX, f"Missing keyword: {kw}"


def test_vocab_contains_operators():
    for op in ["=", "==", "!=", "<", ">", "+", "-", "*", "/"]:
        assert op in VOCAB_INDEX, f"Missing operator: {op}"


def test_vocab_contains_punctuation():
    for p in ["(", ")", ":", ",", ".", "[", "]", "{", "}"]:
        assert p in VOCAB_INDEX, f"Missing punctuation: {p}"


def test_vocab_contains_builtins():
    for b in ["print", "len", "range", "type", "int", "str", "float", "bool", "list"]:
        assert b in VOCAB_INDEX, f"Missing builtin: {b}"


def test_compose_context_node_shape():
    vec = compose_context_node("for", "in", "range", "(")
    assert vec.shape == (D,)


def test_compose_context_node_dtype():
    vec = compose_context_node("if", ":", "pass", "else")
    assert vec.dtype == np.float64


def test_compose_context_node_weights_sum():
    """Weights 0.20+0.35+0.35+0.10=1.0, so a single repeated token should sum to 1.0."""
    vec = compose_context_node("if", "if", "if", "if")
    assert abs(vec.sum() - 1.0) < 1e-10


def test_compose_context_node_unknown_token():
    """Unknown tokens contribute zero weight."""
    vec = compose_context_node("<unk>", "<unk>", "<unk>", "<unk>")
    assert vec.sum() == 0.0


def test_compose_context_node_slot_weights():
    """Verify individual slot weights: left1=0.35."""
    # Only left1 slot is a known token
    vec = compose_context_node("<unk>", "if", "<unk>", "<unk>")
    idx = VOCAB_INDEX["if"]
    assert abs(vec[idx] - 0.35) < 1e-10


def test_compose_context_node_right1_weight():
    """right1 slot weight is 0.35."""
    vec = compose_context_node("<unk>", "<unk>", "for", "<unk>")
    idx = VOCAB_INDEX["for"]
    assert abs(vec[idx] - 0.35) < 1e-10


def test_compose_context_node_left2_weight():
    """left2 slot weight is 0.20."""
    vec = compose_context_node("def", "<unk>", "<unk>", "<unk>")
    idx = VOCAB_INDEX["def"]
    assert abs(vec[idx] - 0.20) < 1e-10


def test_compose_context_node_right2_weight():
    """right2 slot weight is 0.10."""
    vec = compose_context_node("<unk>", "<unk>", "<unk>", "class")
    idx = VOCAB_INDEX["class"]
    assert abs(vec[idx] - 0.10) < 1e-10


def test_generate_code_snippets_returns_list():
    obs = generate_code_snippets(seed=42)
    assert isinstance(obs, list)
    assert len(obs) > 0


def test_generate_code_snippets_tuple_structure():
    obs = generate_code_snippets(seed=42)
    vec, token, category = obs[0]
    assert isinstance(vec, np.ndarray)
    assert isinstance(token, str)
    assert isinstance(category, str)


def test_generate_code_snippets_vector_shape():
    obs = generate_code_snippets(seed=42)
    for vec, _, _ in obs[:10]:
        assert vec.shape == (D,)


def test_generate_code_snippets_categories():
    obs = generate_code_snippets(seed=42)
    categories = {cat for _, _, cat in obs}
    assert "control_flow" in categories
    assert "functions" in categories
    assert "data" in categories
    assert "builtins" in categories


def test_generate_code_snippets_true_tokens_in_vocab():
    obs = generate_code_snippets(seed=42)
    for _, token, _ in obs[:50]:
        assert token in VOCAB_INDEX, f"Token not in vocab: {token}"


def test_generate_code_snippets_different_seeds():
    obs1 = generate_code_snippets(seed=42)
    obs2 = generate_code_snippets(seed=99)
    # Different seeds should produce different ordering
    tokens1 = [t for _, t, _ in obs1[:10]]
    tokens2 = [t for _, t, _ in obs2[:10]]
    assert tokens1 != tokens2
