"""Tests for SP9 Cross-Domain L4 Benchmark (benchmarks/phyre_cross_domain_l4.py)."""
from __future__ import annotations

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.phyre_cross_domain_l4 import (
    PAD_DIM,
    _pad,
    encode_domain_pairs,
    fit_cross_domain_l4,
    score_cross_domain,
    score_l2l3_baseline,
    run_rotation,
)


# ---------------------------------------------------------------------------
# Unit tests: _pad
# ---------------------------------------------------------------------------

class TestPad:
    def test_pad_short_vector(self):
        v = np.array([1.0, 2.0, 3.0])
        result = _pad(v, 14)
        assert result.shape == (14,)
        assert result[0] == pytest.approx(1.0)
        assert result[3] == pytest.approx(0.0)

    def test_pad_exact_length(self):
        v = np.ones(14)
        result = _pad(v, 14)
        assert result.shape == (14,)
        np.testing.assert_array_almost_equal(result, v)

    def test_pad_longer_vector_truncates(self):
        v = np.arange(20, dtype=float)
        result = _pad(v, 14)
        assert result.shape == (14,)
        np.testing.assert_array_almost_equal(result, v[:14])

    def test_pad_produces_pad_dim(self):
        """Padding any domain's native dim produces PAD_DIM output."""
        for native_dim in [9, 10, 12, 14]:  # ARC L2=9, Math L2=10, L3=12, PhyRE L2=14
            v = np.ones(native_dim)
            result = _pad(v, PAD_DIM)
            assert result.shape == (PAD_DIM,), f"native_dim={native_dim}"


# ---------------------------------------------------------------------------
# Unit tests: encode_domain_pairs with math (lightweight)
# ---------------------------------------------------------------------------

class TestEncodeDomainPairsMath:
    @pytest.fixture
    def math_tasks(self):
        from benchmarks.structured_math import generate_tasks
        return generate_tasks(n_per_family=3, seed=0)

    def test_returns_list_of_tuples(self, math_tasks):
        from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder
        l2_enc, l3_enc = MathL2Encoder(), MathL3Encoder()
        pairs = encode_domain_pairs(math_tasks, 'math', l2_enc, l3_enc)
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_pair_dims_are_pad_dim(self, math_tasks):
        from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder
        l2_enc, l3_enc = MathL2Encoder(), MathL3Encoder()
        pairs = encode_domain_pairs(math_tasks, 'math', l2_enc, l3_enc)
        for l2, l3 in pairs:
            assert l2.shape == (PAD_DIM,), f"L2 shape={l2.shape}"
            assert l3.shape == (PAD_DIM,), f"L3 shape={l3.shape}"


# ---------------------------------------------------------------------------
# Unit tests: fit_cross_domain_l4
# ---------------------------------------------------------------------------

class TestFitCrossDomainL4:
    def test_fit_returns_head(self):
        from hpm.agents.l4_generative import L4GenerativeHead
        pairs = [(np.random.rand(14).astype(np.float32),
                  np.random.rand(14).astype(np.float32)) for _ in range(10)]
        head = fit_cross_domain_l4(pairs)
        assert isinstance(head, L4GenerativeHead)

    def test_predict_after_fit(self):
        pairs = [(np.random.rand(14).astype(np.float32),
                  np.random.rand(14).astype(np.float32)) for _ in range(10)]
        head = fit_cross_domain_l4(pairs)
        pred = head.predict(np.ones(14, dtype=np.float32))
        assert pred is not None
        assert pred.shape == (14,)

    def test_predict_none_before_fit(self):
        from hpm.agents.l4_generative import L4GenerativeHead
        head = L4GenerativeHead(feature_dim_in=14, feature_dim_out=14)
        pred = head.predict(np.ones(14, dtype=np.float32))
        assert pred is None

    def test_fit_with_one_pair_does_not_crash(self):
        """Fit with < 2 pairs should be a no-op (predict returns None)."""
        pairs = [(np.ones(14, dtype=np.float32), np.zeros(14, dtype=np.float32))]
        head = fit_cross_domain_l4(pairs)
        # With < 2 pairs, fit is a no-op and predict returns None
        pred = head.predict(np.ones(14, dtype=np.float32))
        assert pred is None


# ---------------------------------------------------------------------------
# Smoke test: full rotation on tiny data (math + phyre -> arc)
# ---------------------------------------------------------------------------

class TestRunRotationSmoke:
    def test_math_phyre_to_arc_returns_valid_accuracy(self):
        """Fit on 5 math + 5 phyre tasks, eval on 3 arc tasks."""
        # Use tiny n_tasks to keep test fast
        result = run_rotation(
            train_domains=['math', 'phyre'],
            test_domain='arc',
            n_per_domain=5,
            seed=0,
        )
        assert 'l2l3' in result
        assert 'cross_domain_l4' in result
        assert 0.0 <= result['l2l3'] <= 1.0, f"l2l3={result['l2l3']}"
        assert 0.0 <= result['cross_domain_l4'] <= 1.0, f"cross_domain_l4={result['cross_domain_l4']}"

    def test_math_arc_to_phyre_returns_valid_accuracy(self):
        result = run_rotation(
            train_domains=['math', 'arc'],
            test_domain='phyre',
            n_per_domain=4,
            seed=1,
        )
        assert 0.0 <= result['l2l3'] <= 1.0
        assert 0.0 <= result['cross_domain_l4'] <= 1.0

    def test_phyre_arc_to_math_returns_valid_accuracy(self):
        result = run_rotation(
            train_domains=['phyre', 'arc'],
            test_domain='math',
            n_per_domain=4,
            seed=2,
        )
        assert 0.0 <= result['l2l3'] <= 1.0
        assert 0.0 <= result['cross_domain_l4'] <= 1.0

    def test_n_train_pairs_positive(self):
        result = run_rotation(
            train_domains=['math', 'phyre'],
            test_domain='arc',
            n_per_domain=5,
            seed=0,
        )
        assert result['n_train_pairs'] > 0

    def test_n_test_positive(self):
        result = run_rotation(
            train_domains=['math', 'phyre'],
            test_domain='arc',
            n_per_domain=5,
            seed=0,
        )
        assert result['n_test'] > 0
