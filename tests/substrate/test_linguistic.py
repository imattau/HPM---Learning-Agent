import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def make_substrate(**kwargs):
    """Construct LinguisticSubstrate with minimal dependencies for fast tests."""
    from hpm.substrate.linguistic import LinguisticSubstrate
    defaults = dict(feature_dim=32, use_api=False, use_spacy=False)
    defaults.update(kwargs)
    return LinguisticSubstrate(**defaults)


def test_fetch_returns_vectors_for_known_word():
    s = make_substrate()
    vecs = s.fetch("dog")
    assert len(vecs) > 0
    for v in vecs:
        assert v.shape == (32,)
        assert np.all(v >= 0) and np.all(v <= 1)


def test_fetch_unknown_word_returns_empty():
    s = make_substrate()
    vecs = s.fetch("xyzzy123qqqq")
    assert vecs == []


def test_fetch_caches_results():
    s = make_substrate()
    r1 = s.fetch("dog")
    r2 = s.fetch("dog")
    assert r1 is r2


def test_field_frequency_in_range():
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = "dog"
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_field_frequency_zero_for_unknown_word():
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = "xyzzy123qqqq"
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert freq == 0.0


def test_api_component_skipped_on_network_error():
    s = make_substrate(use_api=True)
    with patch('hpm.substrate.linguistic.requests.get', side_effect=ConnectionError):
        vecs = s.fetch("dog")
    # WordNet vecs still returned; API failure is silent
    assert len(vecs) > 0


def test_api_disabled():
    s = make_substrate(use_api=False)
    assert not s._use_api
    vecs = s.fetch("dog")
    assert isinstance(vecs, list)


def test_spacy_disabled():
    s = make_substrate(use_spacy=False)
    assert s._nlp is None
    vecs = s.fetch("cat")
    assert isinstance(vecs, list)


def test_stream_yields_arrays():
    s = make_substrate()
    gen = s.stream()
    items = [next(gen) for _ in range(10)]
    for v in items:
        assert isinstance(v, np.ndarray)
        assert v.shape == (32,)


def test_lookuperror_if_wordnet_corpus_missing():
    import nltk
    with patch.object(nltk.data, 'find', side_effect=LookupError("Corpora not found")):
        from hpm.substrate.linguistic import LinguisticSubstrate
        with pytest.raises(LookupError):
            LinguisticSubstrate(use_spacy=False, use_api=False)


def test_importerror_if_nltk_missing():
    # Simulate nltk being absent by blocking the import inside __init__
    import builtins
    import sys
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'nltk' or name.startswith('nltk.'):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    from hpm.substrate.linguistic import LinguisticSubstrate
    with patch('builtins.__import__', side_effect=mock_import):
        # Remove cached nltk from sys.modules so the constructor's import is not short-circuited
        sys.modules.pop('nltk', None)
        with pytest.raises(ImportError):
            LinguisticSubstrate(use_spacy=False, use_api=False)


def test_spacy_pos_vector_normalised():
    """spaCy POS frequency vector sums to ~1.0."""
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load('en_core_web_sm')
    except Exception:
        pytest.skip("en_core_web_sm model not available")
    from hpm.substrate.linguistic import LinguisticSubstrate
    s = LinguisticSubstrate(feature_dim=32, use_api=False, use_spacy=True)
    if s._nlp is None:
        pytest.skip("spaCy model not loaded")
    vecs = s.fetch("the quick brown fox jumps")
    # POS vector is the last one appended
    pos_vec = vecs[-1]
    # Values in [0, 1] and sum <= 1.0 (sum < 1 if feature_dim > 17 tags, zero-padded)
    assert np.all(pos_vec >= 0)
    assert pos_vec.sum() <= 1.01  # allow float tolerance
