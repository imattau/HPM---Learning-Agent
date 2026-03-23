"""Tests for SP10 Delta Alignment Benchmark (benchmarks/phyre_delta_alignment.py)."""
from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.phyre_delta_alignment import (
    PAD_DIM,
    _pad,
    compute_delta_pairs,
    fit_domain_matrix,
    procrustes_align,
    score_with_anchor,
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

    def test_pad_longer_vector_truncates(self):
        v = np.arange(20, dtype=float)
        result = _pad(v, 14)
        assert result.shape == (14,)
        np.testing.assert_array_almost_equal(result, v[:14])


# ---------------------------------------------------------------------------
# Unit tests: fit_domain_matrix
# ---------------------------------------------------------------------------

class TestFitDomainMatrix:
    def test_shape_is_14x14(self):
        """M_d shape must be (14, 14) for each domain."""
        rng = np.random.default_rng(0)
        delta_pairs = [
            (rng.standard_normal(PAD_DIM), rng.standard_normal(PAD_DIM))
            for _ in range(50)
        ]
        M = fit_domain_matrix(delta_pairs)
        assert M.shape == (PAD_DIM, PAD_DIM), f"Expected ({PAD_DIM},{PAD_DIM}), got {M.shape}"

    def test_shape_with_math_domain(self):
        """Shape is (14,14) regardless of native encoder dimension."""
        rng = np.random.default_rng(1)
        # Simulate Math: L2=10-dim, L3=12-dim, padded to 14
        delta_pairs = [
            (
                np.pad(rng.standard_normal(10), (0, 4)),
                np.pad(rng.standard_normal(12), (0, 2)),
            )
            for _ in range(30)
        ]
        M = fit_domain_matrix(delta_pairs)
        assert M.shape == (PAD_DIM, PAD_DIM)

    def test_empty_delta_pairs_returns_identity(self):
        M = fit_domain_matrix([])
        assert M.shape == (PAD_DIM, PAD_DIM)
        np.testing.assert_array_almost_equal(M, np.eye(PAD_DIM))

    def test_regression_reduces_error(self):
        """Fitted M should predict better than identity on training data."""
        rng = np.random.default_rng(42)
        # True map: rotate + scale
        true_M = rng.standard_normal((PAD_DIM, PAD_DIM)) * 0.5
        X = rng.standard_normal((100, PAD_DIM))
        Y = X @ true_M.T + rng.standard_normal((100, PAD_DIM)) * 0.01

        delta_pairs = list(zip(X, Y))
        M = fit_domain_matrix(delta_pairs, alpha=0.001)

        # M should be close to true_M
        error = np.linalg.norm(M - true_M)
        assert error < 2.0, f"Regression error too large: {error}"


# ---------------------------------------------------------------------------
# Unit tests: procrustes_align
# ---------------------------------------------------------------------------

class TestProcrustesAlign:
    def _get_R(self, M1: np.ndarray, M2: np.ndarray):
        """Extract R from procrustes computation for testing."""
        U, s, Vt = np.linalg.svd(M2 @ M1.T)
        V = Vt.T
        d = np.sign(np.linalg.det(V @ U.T))
        if d == 0:
            d = 1.0
        D = np.diag([1.0] * (PAD_DIM - 1) + [float(d)])
        R = V @ D @ U.T
        return R

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        M1 = rng.standard_normal((PAD_DIM, PAD_DIM))
        M2 = rng.standard_normal((PAD_DIM, PAD_DIM))
        M_shared = procrustes_align(M1, M2)
        assert M_shared.shape == (PAD_DIM, PAD_DIM)

    def test_R_is_orthogonal(self):
        """Assert R @ R.T - I < 1e-10."""
        rng = np.random.default_rng(1)
        M1 = rng.standard_normal((PAD_DIM, PAD_DIM))
        M2 = rng.standard_normal((PAD_DIM, PAD_DIM))
        R = self._get_R(M1, M2)
        residual = np.linalg.norm(R @ R.T - np.eye(PAD_DIM))
        assert residual < 1e-10, f"R is not orthogonal: residual={residual}"

    def test_det_R_is_plus_one(self):
        """Assert det(R) ≈ +1.0 (proper rotation, not reflection)."""
        rng = np.random.default_rng(2)
        M1 = rng.standard_normal((PAD_DIM, PAD_DIM))
        M2 = rng.standard_normal((PAD_DIM, PAD_DIM))
        R = self._get_R(M1, M2)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, f"det(R)={det}, expected +1.0"

    def test_shared_is_average(self):
        """M_shared = (M1 + R^T @ M2) / 2."""
        rng = np.random.default_rng(3)
        M1 = rng.standard_normal((PAD_DIM, PAD_DIM))
        M2 = rng.standard_normal((PAD_DIM, PAD_DIM))
        M_shared = procrustes_align(M1, M2)
        R = self._get_R(M1, M2)
        expected = (M1 + R.T @ M2) / 2.0
        np.testing.assert_array_almost_equal(M_shared, expected, decimal=10)

    def test_det_check_multiple_random_matrices(self):
        """det(R) == +1 across many random matrix pairs."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            M1 = rng.standard_normal((PAD_DIM, PAD_DIM))
            M2 = rng.standard_normal((PAD_DIM, PAD_DIM))
            R = self._get_R(M1, M2)
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-8, f"det(R)={det}"


# ---------------------------------------------------------------------------
# Unit tests: compute_delta_pairs
# ---------------------------------------------------------------------------

class TestComputeDeltaPairs:
    @pytest.fixture
    def math_tasks(self):
        from benchmarks.structured_math import generate_tasks
        return generate_tasks(n_per_family=3, seed=0)

    def test_pairwise_count(self, math_tasks):
        """Assert pairwise delta count = N*(N-1) for N encoded train pairs."""
        from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder
        l2_enc, l3_enc = MathL2Encoder(), MathL3Encoder()

        # Count how many valid train pairs we'd get
        from benchmarks.phyre_cross_domain_l4 import encode_domain_pairs, PAD_DIM as PDIM
        encoded_pairs = encode_domain_pairs(math_tasks, 'math', l2_enc, l3_enc)
        N = len(encoded_pairs)

        delta_pairs = compute_delta_pairs(math_tasks, 'math', l2_enc, l3_enc)
        assert len(delta_pairs) == N * (N - 1), (
            f"Expected {N*(N-1)} delta pairs for N={N}, got {len(delta_pairs)}"
        )

    def test_delta_shape(self, math_tasks):
        """Each delta must be (PAD_DIM,)."""
        from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder
        l2_enc, l3_enc = MathL2Encoder(), MathL3Encoder()
        delta_pairs = compute_delta_pairs(math_tasks, 'math', l2_enc, l3_enc)
        assert len(delta_pairs) > 0
        for dl2, dl3 in delta_pairs:
            assert dl2.shape == (PAD_DIM,), f"ΔL2 shape={dl2.shape}"
            assert dl3.shape == (PAD_DIM,), f"ΔL3 shape={dl3.shape}"

    def test_empty_tasks_returns_empty(self):
        from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder
        l2_enc, l3_enc = MathL2Encoder(), MathL3Encoder()
        delta_pairs = compute_delta_pairs([], 'math', l2_enc, l3_enc)
        assert delta_pairs == []


# ---------------------------------------------------------------------------
# Smoke test: full rotation on tiny data
# ---------------------------------------------------------------------------

class TestRunRotationSmoke:
    def test_math_phyre_to_arc_accuracy_in_range(self):
        """Fit on Math + PhyRE, eval on ARC; accuracy must be in [0, 1]."""
        result = run_rotation(
            train_domains=['math', 'phyre'],
            test_domain='arc',
            n_per_family=5,
            seed=0,
        )
        assert 'l2l3' in result
        assert 'delta_alignment' in result
        assert 0.0 <= result['l2l3'] <= 1.0, f"l2l3={result['l2l3']}"
        assert 0.0 <= result['delta_alignment'] <= 1.0, f"delta_alignment={result['delta_alignment']}"

    def test_math_arc_to_phyre_accuracy_in_range(self):
        result = run_rotation(
            train_domains=['math', 'arc'],
            test_domain='phyre',
            n_per_family=4,
            seed=1,
        )
        assert 0.0 <= result['l2l3'] <= 1.0
        assert 0.0 <= result['delta_alignment'] <= 1.0

    def test_phyre_arc_to_math_accuracy_in_range(self):
        result = run_rotation(
            train_domains=['phyre', 'arc'],
            test_domain='math',
            n_per_family=4,
            seed=2,
        )
        assert 0.0 <= result['l2l3'] <= 1.0
        assert 0.0 <= result['delta_alignment'] <= 1.0

    def test_n_train_pairs_positive(self):
        result = run_rotation(
            train_domains=['math', 'phyre'],
            test_domain='arc',
            n_per_family=5,
            seed=0,
        )
        assert result['n_train_pairs'] > 0

    def test_n_test_positive(self):
        result = run_rotation(
            train_domains=['math', 'phyre'],
            test_domain='arc',
            n_per_family=5,
            seed=0,
        )
        assert result['n_test'] > 0
