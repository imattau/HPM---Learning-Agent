import numpy as np
import pytest
from hpm.substrate.local_file import LocalFileSubstrate
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def text_dir(tmp_path):
    (tmp_path / "doc1.txt").write_text("the cat sat on the mat")
    (tmp_path / "doc2.txt").write_text("dogs and cats are pets")
    (tmp_path / "doc3.txt").write_text("machine learning is useful")
    return str(tmp_path)


@pytest.fixture
def substrate(text_dir):
    return LocalFileSubstrate(text_dir, feature_dim=16)


def test_fetch_returns_vectors(substrate):
    vecs = substrate.fetch("cat")
    assert len(vecs) > 0
    assert all(v.shape == (16,) for v in vecs)


def test_fetch_no_match_returns_all_docs(substrate):
    vecs = substrate.fetch("nonexistent_xyz_term")
    # Falls back to all documents
    assert len(vecs) == 3


def test_stream_yields_vectors(substrate):
    gen = substrate.stream()
    v = next(gen)
    assert v.shape == (16,)


def test_field_frequency_returns_float_in_range(substrate):
    pattern = GaussianPattern(mu=np.zeros(16), sigma=np.eye(16))
    freq = substrate.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_caching_returns_same_result(substrate):
    vecs1 = substrate.fetch("cat")
    vecs2 = substrate.fetch("cat")
    assert len(vecs1) == len(vecs2)
    for v1, v2 in zip(vecs1, vecs2):
        assert np.allclose(v1, v2)
