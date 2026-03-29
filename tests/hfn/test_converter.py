"""Tests for the Converter base class."""

import numpy as np
import pytest
from hfn.converter import Converter


class FixedConverter(Converter):
    """Subclass that returns fixed arrays."""
    def encode(self, raw: str, D: int) -> list[np.ndarray]:
        # Return a single zero vector of dimension D
        return [np.zeros(D)]


class OneHotConverter(Converter):
    """Subclass that encodes a single integer token as one-hot."""
    def encode(self, raw: str, D: int) -> list[np.ndarray]:
        try:
            idx = int(raw)
        except ValueError:
            return []
        if idx < 0 or idx >= D:
            return []
        vec = np.zeros(D)
        vec[idx] = 1.0
        return [vec]


def test_base_converter_raises_not_implemented():
    c = Converter()
    with pytest.raises(NotImplementedError):
        c.encode("test", 10)


def test_fixed_converter_returns_list():
    c = FixedConverter()
    result = c.encode("anything", 10)
    assert isinstance(result, list)
    assert len(result) == 1


def test_fixed_converter_returns_correct_shape():
    c = FixedConverter()
    D = 16
    result = c.encode("test", D)
    assert result[0].shape == (D,)


def test_fixed_converter_returns_ndarray():
    c = FixedConverter()
    result = c.encode("test", 8)
    assert isinstance(result[0], np.ndarray)


def test_one_hot_converter_valid_token():
    c = OneHotConverter()
    D = 10
    result = c.encode("3", D)
    assert len(result) == 1
    assert result[0].shape == (D,)
    assert result[0][3] == 1.0
    assert result[0].sum() == 1.0


def test_one_hot_converter_invalid_token_returns_empty():
    c = OneHotConverter()
    result = c.encode("not_an_int", 10)
    assert result == []


def test_one_hot_converter_out_of_range_returns_empty():
    c = OneHotConverter()
    result = c.encode("99", 10)
    assert result == []


def test_converter_multiple_outputs():
    """Subclass can return multiple arrays per raw string."""
    class MultiConverter(Converter):
        def encode(self, raw: str, D: int) -> list[np.ndarray]:
            return [np.ones(D), np.zeros(D)]

    c = MultiConverter()
    result = c.encode("test", 5)
    assert len(result) == 2
    assert result[0].shape == (5,)
    assert result[1].shape == (5,)
