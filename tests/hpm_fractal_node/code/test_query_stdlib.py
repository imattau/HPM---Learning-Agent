"""Tests for QueryStdlib."""

import numpy as np
import pytest
from hpm_fractal_node.code.query_stdlib import QueryStdlib
from hpm_fractal_node.code.code_loader import VOCAB, VOCAB_INDEX, VOCAB_SIZE


def _gap_mu_for(token: str) -> np.ndarray:
    """Create a gap_mu with argmax at the given token's index."""
    mu = np.zeros(VOCAB_SIZE)
    if token in VOCAB_INDEX:
        mu[VOCAB_INDEX[token]] = 1.0
    return mu


def test_query_stdlib_instantiation():
    q = QueryStdlib()
    assert q.max_results == 20


def test_query_stdlib_custom_max_results():
    q = QueryStdlib(max_results=5)
    assert q.max_results == 5


def test_query_stdlib_fetch_returns_list():
    q = QueryStdlib(max_results=5)
    mu = _gap_mu_for("for")
    result = q.fetch(mu)
    assert isinstance(result, list)


def test_query_stdlib_fetch_result_strings():
    q = QueryStdlib(max_results=5)
    mu = _gap_mu_for("for")
    result = q.fetch(mu)
    for item in result:
        assert isinstance(item, str)


def test_query_stdlib_fetch_respects_max_results():
    max_n = 3
    q = QueryStdlib(max_results=max_n)
    mu = _gap_mu_for("for")
    result = q.fetch(mu)
    assert len(result) <= max_n


def test_query_stdlib_fetch_oob_index_returns_empty():
    """Argmax index beyond VOCAB length returns empty list."""
    q = QueryStdlib(max_results=5)
    mu = np.zeros(VOCAB_SIZE + 10)
    mu[-1] = 1.0
    result = q.fetch(mu)
    assert result == []


def test_query_stdlib_sig_prefix():
    """Signature strings start with 'sig: '."""
    q = QueryStdlib(max_results=20)
    # Use "def" as target — should find many function definitions
    mu = _gap_mu_for("def")
    result = q.fetch(mu)
    sig_results = [r for r in result if r.startswith("sig: ")]
    ctx_results = [r for r in result if not r.startswith("sig: ")]
    # At least some results expected for "def"
    # (stdlib has many def statements)
    # All items are either sig: or context window
    for r in result:
        assert r.startswith("sig: ") or len(r.split()) == 4, \
            f"Unexpected format: {r!r}"


def test_query_stdlib_context_window_format():
    """Non-sig results should have exactly 4 space-separated tokens."""
    q = QueryStdlib(max_results=10)
    mu = _gap_mu_for("for")
    result = q.fetch(mu)
    for r in result:
        if not r.startswith("sig: "):
            parts = r.split()
            assert len(parts) == 4, f"Context window should have 4 tokens: {r!r}"


def test_query_stdlib_no_duplicates():
    """Results should not contain duplicates."""
    q = QueryStdlib(max_results=20)
    mu = _gap_mu_for("if")
    result = q.fetch(mu)
    assert len(result) == len(set(result))


def test_query_stdlib_known_common_token():
    """'for' appears in stdlib — should find results."""
    q = QueryStdlib(max_results=10)
    mu = _gap_mu_for("for")
    result = q.fetch(mu)
    # stdlib certainly has 'for' loops
    assert len(result) > 0
