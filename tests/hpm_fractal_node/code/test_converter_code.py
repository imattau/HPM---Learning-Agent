"""Tests for ConverterCode."""

import numpy as np
import pytest
from hpm_fractal_node.code.converter_code import ConverterCode
from hpm_fractal_node.code.code_loader import VOCAB_INDEX, VOCAB_SIZE, D


def test_converter_code_instantiation():
    c = ConverterCode()


def test_encode_context_window_returns_list():
    c = ConverterCode()
    result = c.encode("for in range (", D)
    assert isinstance(result, list)


def test_encode_context_window_single_vector():
    c = ConverterCode()
    result = c.encode("for in range (", D)
    assert len(result) == 1


def test_encode_context_window_shape():
    c = ConverterCode()
    result = c.encode("if == : else", D)
    assert result[0].shape == (D,)


def test_encode_context_window_dtype():
    c = ConverterCode()
    result = c.encode("def ( ) :", D)
    assert result[0].dtype == np.float64


def test_encode_context_window_unk_tokens():
    """'<unk>' tokens contribute zero weight."""
    c = ConverterCode()
    result = c.encode("<unk> <unk> <unk> <unk>", D)
    assert len(result) == 1
    assert result[0].sum() == 0.0


def test_encode_context_window_wrong_token_count():
    """Wrong number of tokens returns empty list."""
    c = ConverterCode()
    result = c.encode("for in", D)
    assert result == []


def test_encode_context_window_five_tokens():
    """5 tokens returns empty list."""
    c = ConverterCode()
    result = c.encode("for in range ( )", D)
    assert result == []


def test_encode_sig_returns_list():
    c = ConverterCode()
    result = c.encode("sig: def for range :", D)
    assert isinstance(result, list)


def test_encode_sig_single_vector():
    c = ConverterCode()
    result = c.encode("sig: def for range :", D)
    assert len(result) == 1


def test_encode_sig_shape():
    c = ConverterCode()
    result = c.encode("sig: def for range :", D)
    assert result[0].shape == (D,)


def test_encode_sig_dtype():
    c = ConverterCode()
    result = c.encode("sig: def for range :", D)
    assert result[0].dtype == np.float64


def test_encode_sig_sums_to_one_for_single_token():
    """A sig with a single known token should produce a one-hot vector."""
    c = ConverterCode()
    result = c.encode("sig: for", D)
    assert len(result) == 1
    assert abs(result[0].sum() - 1.0) < 1e-10
    assert result[0][VOCAB_INDEX["for"]] == 1.0


def test_encode_sig_bag_of_tokens():
    """Bag of 2 equal tokens: each gets weight 0.5."""
    c = ConverterCode()
    result = c.encode("sig: for if", D)
    vec = result[0]
    assert abs(vec[VOCAB_INDEX["for"]] - 0.5) < 1e-10
    assert abs(vec[VOCAB_INDEX["if"]] - 0.5) < 1e-10
    assert abs(vec.sum() - 1.0) < 1e-10


def test_encode_sig_skips_unk():
    """<unk> tokens are skipped in sig encoding."""
    c = ConverterCode()
    result_with_unk = c.encode("sig: for <unk>", D)
    result_without = c.encode("sig: for", D)
    np.testing.assert_array_almost_equal(result_with_unk[0], result_without[0])


def test_encode_sig_all_unk_returns_empty():
    """All OOV tokens returns empty list."""
    c = ConverterCode()
    result = c.encode("sig: <unk> <unk>", D)
    assert result == []


def test_encode_sig_prefix_stripped():
    """'sig: ' prefix is stripped before encoding."""
    c = ConverterCode()
    # "sig: for" and "for" as a 4-token context have different encodings
    sig_result = c.encode("sig: for", D)
    assert len(sig_result) == 1
    # The sig result is one-hot at "for"
    assert sig_result[0][VOCAB_INDEX["for"]] == 1.0


def test_encode_context_nonzero_for_known_tokens():
    """Context window with known tokens produces nonzero vector."""
    c = ConverterCode()
    result = c.encode("for in range (", D)
    assert result[0].sum() > 0.0
