import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from hpm.substrate.wikipedia import WikipediaSubstrate
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def substrate():
    return WikipediaSubstrate(feature_dim=32)


def _mock_response(status: int, extract: str = ""):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {'title': 'Test', 'extract': extract}
    return mock


def test_fetch_returns_vectors(substrate):
    with patch('requests.get', return_value=_mock_response(200, "A cat is a small furry animal that purrs.")):
        vecs = substrate.fetch("cat")
    assert len(vecs) > 0
    assert all(v.shape == (32,) for v in vecs)


def test_fetch_404_returns_empty(substrate):
    with patch('requests.get', return_value=_mock_response(404)):
        vecs = substrate.fetch("nonexistent_page_xyz_abc")
    assert vecs == []


def test_field_frequency_returns_float_in_range(substrate):
    pattern = GaussianPattern(mu=np.zeros(32), sigma=np.eye(32))
    with patch('requests.get', return_value=_mock_response(200, "test content here")):
        freq = substrate.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_caching_avoids_repeated_requests(substrate):
    with patch('requests.get', return_value=_mock_response(200, "A dog is a domestic animal.")) as mock_get:
        substrate.fetch("dog")
        substrate.fetch("dog")
    assert mock_get.call_count == 1
