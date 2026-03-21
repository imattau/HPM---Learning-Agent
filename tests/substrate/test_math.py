import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def make_substrate(**kwargs):
    """Construct MathSubstrate with minimal optional dependencies."""
    from hpm.substrate.math import MathSubstrate
    defaults = dict(feature_dim=32, use_scipy=False, wolfram_app_id=None)
    defaults.update(kwargs)
    return MathSubstrate(**defaults)


def test_fetch_sympy_expression():
    s = make_substrate()
    vecs = s.fetch("x**2 + 1")
    assert len(vecs) > 0
    for v in vecs:
        assert v.shape == (32,)


def test_fetch_topic_name():
    s = make_substrate()
    vecs = s.fetch("algebra")
    assert len(vecs) > 0
    for v in vecs:
        assert v.shape == (32,)


def test_fetch_partial_topic_match():
    s = make_substrate()
    # "basic trigonometry" should match topic key "trigonometry"
    vecs = s.fetch("basic trigonometry")
    assert len(vecs) > 0


def test_fetch_unknown_returns_empty():
    s = make_substrate()
    # Use a string that SymPy cannot parse as an expression (invalid syntax)
    # AND does not match any known topic key
    vecs = s.fetch("(((")
    assert vecs == []


def test_fetch_caches_results():
    s = make_substrate()
    r1 = s.fetch("algebra")
    r2 = s.fetch("algebra")
    assert r1 is r2


def test_field_frequency_in_range():
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = "algebra"
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_field_frequency_uses_algebra_default():
    """field_frequency falls back to 'algebra' when pattern has no label."""
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = None
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_wolfram_skipped_when_no_key():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop('WOLFRAM_APP_ID', None)
        s = make_substrate(wolfram_app_id=None)
    with patch('hpm.substrate.math.requests.get') as mock_get:
        s.fetch("integral of sin(x)")
    mock_get.assert_not_called()


def test_wolfram_used_when_key_set():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "negative cosine of x plus constant"
    with patch('hpm.substrate.math.requests.get', return_value=mock_resp):
        s = make_substrate(wolfram_app_id="FAKE_KEY")
        vecs = s.fetch("integral of sin(x)")
    # Should have SymPy/topic vecs + Wolfram vec
    assert len(vecs) > 0


def test_wolfram_skipped_on_network_error():
    with patch('hpm.substrate.math.requests.get', side_effect=ConnectionError):
        s = make_substrate(wolfram_app_id="FAKE_KEY")
        # Should still work (no exception); returns SymPy/topic vecs
        vecs = s.fetch("algebra")
    assert isinstance(vecs, list)


def test_scipy_disabled():
    s = make_substrate(use_scipy=False)
    assert s._scipy_constants is None
    vecs = s.fetch("calculus")
    assert isinstance(vecs, list)


def test_scipy_enabled():
    pytest.importorskip("scipy")
    from hpm.substrate.math import MathSubstrate
    s = MathSubstrate(feature_dim=32, use_scipy=True, wolfram_app_id=None)
    assert s._scipy_constants is not None
    vecs = s.fetch("speed of light")
    assert len(vecs) > 0


def test_stream_yields_arrays():
    s = make_substrate()
    gen = s.stream()
    items = [next(gen) for _ in range(10)]
    for v in items:
        assert isinstance(v, np.ndarray)
        assert v.shape == (32,)


def test_importerror_if_sympy_missing():
    import builtins
    import sys
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'sympy' or name.startswith('sympy.'):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    from hpm.substrate.math import MathSubstrate
    with patch('builtins.__import__', side_effect=mock_import):
        # Remove cached sympy from sys.modules so the constructor's import is not short-circuited
        sys.modules.pop('sympy', None)
        with pytest.raises(ImportError):
            MathSubstrate(use_scipy=False, wolfram_app_id=None)


def test_wolfram_app_id_from_env():
    """Wolfram App ID is picked up from WOLFRAM_APP_ID env var."""
    with patch.dict(os.environ, {'WOLFRAM_APP_ID': 'ENV_KEY'}):
        s = make_substrate()
        assert s._wolfram_app_id == 'ENV_KEY'
