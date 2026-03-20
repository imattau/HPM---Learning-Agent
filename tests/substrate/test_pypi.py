import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from hpm.substrate.pypi import PyPISubstrate
from hpm.patterns.gaussian import GaussianPattern


FAKE_PYPI_RESPONSE = {
    "info": {
        "name": "numpy",
        "summary": "Fundamental package for array computing in Python.",
        "description": "NumPy is the fundamental package for scientific computing with Python.",
        "keywords": "numpy array scientific computing",
        "version": "1.26.0",
    }
}


def _mock_fetch_package(url, *args, **kwargs):
    mock = MagicMock()
    mock.read.return_value = __import__('json').dumps(FAKE_PYPI_RESPONSE).encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_pypi_substrate_initialises_with_seed_packages():
    sub = PyPISubstrate(seed_packages=["numpy"])
    assert sub.seed_packages == ["numpy"]


def test_fetch_returns_list_of_arrays():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        result = sub.fetch("array computing")
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], np.ndarray)


def test_fetch_returns_empty_for_no_matching_packages():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        result = sub.fetch("zzz_no_match_xyz")
    assert isinstance(result, list)


def test_field_frequency_returns_float_in_unit_interval():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        # feature_dim must equal _VECTOR_DIM (64)
        pattern = GaussianPattern(mu=np.zeros(64), sigma=np.eye(64))
        freq = sub.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_field_frequency_raises_on_dimension_mismatch():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        pattern = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))  # wrong dim
        with pytest.raises(ValueError, match="feature_dim"):
            sub.field_frequency(pattern)


def test_stream_yields_arrays():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        items = list(sub.stream())
    assert len(items) >= 1
    assert isinstance(items[0], np.ndarray)


def test_caching_avoids_duplicate_requests():
    call_count = 0

    def counting_fetch(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_fetch_package(url)

    with patch("urllib.request.urlopen", side_effect=counting_fetch):
        sub = PyPISubstrate(seed_packages=["numpy"], cache=True)
        sub.fetch("array")
        sub.fetch("computing")  # second fetch — should use cache
    assert call_count == 1  # only one HTTP call despite two fetches


def test_no_caching_makes_multiple_requests():
    call_count = 0

    def counting_fetch(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_fetch_package(url)

    with patch("urllib.request.urlopen", side_effect=counting_fetch):
        sub = PyPISubstrate(seed_packages=["numpy"], cache=False)
        sub.fetch("array")
        sub.fetch("computing")
    assert call_count == 2


def test_pypi_substrate_satisfies_external_substrate_protocol():
    """Duck-type check: PyPISubstrate has fetch, field_frequency, stream."""
    sub = PyPISubstrate(seed_packages=[])
    assert callable(sub.fetch)
    assert callable(sub.field_frequency)
    assert callable(sub.stream)
