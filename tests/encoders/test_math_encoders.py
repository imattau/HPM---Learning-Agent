"""Tests for SP5 math encoders and structured math benchmark."""
import numpy as np
import sympy
from sympy import symbols, expand, factor, simplify, diff, integrate
from sympy import diff as sym_diff
from sympy import Integer as SInt

x = symbols('x')


# ---------------------------------------------------------------------------
# Task 1: MathL1Encoder
# ---------------------------------------------------------------------------

def test_math_l1_encoder_shape():
    from hpm.encoders.math_encoders import MathL1Encoder
    enc = MathL1Encoder()
    assert enc.feature_dim == 14
    assert enc.max_steps_per_obs == 1
    obs = (x**2 + 2*x + 1, x**2 - 1)
    vecs = enc.encode(obs, epistemic=None)
    assert len(vecs) == 1
    assert vecs[0].shape == (14,)


def test_math_l1_encoder_values():
    from hpm.encoders.math_encoders import MathL1Encoder
    enc = MathL1Encoder()
    # constant polynomial: x^2 + 2x + 1, output: (x+1)^2 = x^2 + 2x + 1
    obs = (x**2 + 2*x + 1, x**2 + 2*x + 1)
    vecs = enc.encode(obs, epistemic=None)
    vec = vecs[0]
    # sign_lead_in and sign_lead_out both positive
    assert vec[10] == 1.0  # sign_lead_in
    assert vec[11] == 1.0  # sign_lead_out


def test_poly_coeffs_non_polynomial():
    from hpm.encoders.math_encoders import poly_coeffs
    # non-polynomial: sin(x) -> should return zeros
    import sympy as sp
    result = poly_coeffs(sp.sin(x), x)
    assert result.shape == (5,)
    assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# Task 2: MathL2Encoder
# ---------------------------------------------------------------------------

def test_math_l2_encoder_shape():
    from hpm.encoders.math_encoders import MathL2Encoder
    enc = MathL2Encoder()
    assert enc.feature_dim == 10
    assert enc.max_steps_per_obs == 1
    obs = (x**2 - 1, (x + 1) * (x - 1))
    vecs = enc.encode(obs, epistemic=None)
    assert len(vecs) == 1
    assert vecs[0].shape == (10,)


def test_math_l2_encoder_factored_output():
    from hpm.encoders.math_encoders import MathL2Encoder
    enc = MathL2Encoder()
    # (x^2 - 1) -> (x+1)(x-1): output is factored
    obs = (x**2 - 1, (x + 1) * (x - 1))
    vec = enc.encode(obs, epistemic=None)[0]
    # index 4 = is_factored_out
    assert vec[4] == 1.0


def test_math_l2_encoder_epistemic_threading():
    from hpm.encoders.math_encoders import MathL2Encoder
    enc = MathL2Encoder()
    obs = (x**2 - 1, (x + 1) * (x - 1))
    vec_no_ep = enc.encode(obs, epistemic=None)[0]
    vec_with_ep = enc.encode(obs, epistemic=(0.5, 0.3))[0]
    # With epistemic, indices 7 and 8 should reflect l1_weight and l1_loss
    assert vec_with_ep[7] == 0.5  # l1_weight
    assert vec_with_ep[8] == 0.3  # l1_loss
    # Without epistemic, both should be 0.0
    assert vec_no_ep[7] == 0.0
    assert vec_no_ep[8] == 0.0


# ---------------------------------------------------------------------------
# Task 3: MathL3Encoder
# ---------------------------------------------------------------------------

def test_math_l3_encoder_shape():
    from hpm.encoders.math_encoders import MathL3Encoder
    enc = MathL3Encoder()
    assert enc.feature_dim == 12
    assert enc.max_steps_per_obs == 1
    obs = (x**3, sym_diff(x**3, x))
    vecs = enc.encode(obs, epistemic=None)
    assert len(vecs) == 1
    assert vecs[0].shape == (12,)


def test_math_l3_encoder_differentiate():
    from hpm.encoders.math_encoders import MathL3Encoder
    enc = MathL3Encoder()
    # diff(x^3) = 3x^2: degree drops from 3 to 2
    obs = (x**3, sym_diff(x**3, x))
    vec = enc.encode(obs, epistemic=None)[0]
    # index 5 = deg_decreased
    assert vec[5] == 1.0
    # index 6 = deg_unchanged
    assert vec[6] == 0.0
    # index 0 = deg_delta / 4: (2 - 3) / 4 = -0.25
    assert abs(vec[0] - (-0.25)) < 1e-6


def test_math_l3_encoder_expand():
    from hpm.encoders.math_encoders import MathL3Encoder, is_factored_form
    from sympy import expand as sym_expand
    enc = MathL3Encoder()
    # Test that expand of a factored form produces deg_unchanged = 1
    # (x+2)*(x+3) -> x^2 + 5x + 6: same degree
    obs = ((x + 2) * (x + 3), sym_expand((x + 2) * (x + 3)))
    vec = enc.encode(obs, epistemic=None)[0]
    # index 6 = deg_unchanged (degree stays 2)
    assert vec[6] == 1.0
    # factor_delta should be <= 0 (output has same or fewer factors)
    # index 2 = factor_delta / 10
    assert vec[2] <= 0.0


def test_math_l3_encoder_epistemic_threading():
    from hpm.encoders.math_encoders import MathL3Encoder
    enc = MathL3Encoder()
    obs = (x**2, x**2)
    vec = enc.encode(obs, epistemic=(0.8, 0.1))[0]
    # index 10 = l2_weight, index 11 = l2_loss
    assert vec[10] == 0.8
    assert vec[11] == 0.1


# ---------------------------------------------------------------------------
# Task 4: generate_tasks
# ---------------------------------------------------------------------------

def test_generate_tasks_count():
    from benchmarks.structured_math import generate_tasks
    tasks = generate_tasks(n_per_family=5, seed=42)
    # Might be fewer than 25 due to no-op filtering, but must be >0
    assert len(tasks) > 0
    assert len(tasks) <= 25


def test_generate_tasks_structure():
    from benchmarks.structured_math import generate_tasks
    tasks = generate_tasks(n_per_family=5, seed=42)
    for task in tasks:
        assert 'family' in task
        assert task['family'] in {'expand', 'factor', 'simplify', 'differentiate', 'integrate'}
        assert 'train' in task
        assert isinstance(task['train'], list)
        assert 'test_input' in task
        assert 'test_output' in task
        assert 'candidates' in task
        # 5 candidates: 1 correct + 4 distractors
        assert len(task['candidates']) == 5


def test_generate_tasks_no_duplicate_correct():
    """No candidate other than test_output should equal test_output."""
    from benchmarks.structured_math import generate_tasks
    tasks = generate_tasks(n_per_family=5, seed=42)
    for task in tasks:
        correct = task['test_output']
        distractors = [c for c in task['candidates'] if c is not correct]
        for d in distractors:
            diff_expr = sympy.simplify(correct - d)
            assert diff_expr != 0, f"Distractor equals correct answer in family {task['family']}"


# ---------------------------------------------------------------------------
# Task 5: run_benchmark (smoke test)
# ---------------------------------------------------------------------------

def test_benchmark_smoke():
    from benchmarks.structured_math import generate_tasks, run_benchmark
    tasks = generate_tasks(n_per_family=3, seed=0)
    for condition in ['flat', 'l1_only', 'l2_only', 'full']:
        acc = run_benchmark(tasks, condition)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0, f"condition={condition} returned {acc}"


# ---------------------------------------------------------------------------
# Task 6: Edge cases and integration tests
# ---------------------------------------------------------------------------

def test_l1_zero_polynomial():
    from hpm.encoders.math_encoders import MathL1Encoder
    enc = MathL1Encoder()
    # Zero polynomial: both in and out are 0
    obs = (sympy.Integer(0), sympy.Integer(0))
    vecs = enc.encode(obs, epistemic=None)
    assert vecs[0].shape == (14,)
    # All coefficient features should be zero
    assert np.all(vecs[0][:10] == 0.0)


def test_l1_constant_polynomial():
    from hpm.encoders.math_encoders import MathL1Encoder
    enc = MathL1Encoder()
    # diff of constant: output is 0
    obs = (sympy.Integer(5), sympy.Integer(0))
    vecs = enc.encode(obs, epistemic=None)
    assert vecs[0].shape == (14,)


def test_l2_rational_output():
    from hpm.encoders.math_encoders import MathL2Encoder
    enc = MathL2Encoder()
    # 1/x is rational
    obs = (x**2, sympy.Integer(1) / x)
    vecs = enc.encode(obs, epistemic=None)
    vec = vecs[0]
    # index 5 = has_rational_out
    assert vec[5] == 1.0


def test_l3_degree_unchanged():
    from hpm.encoders.math_encoders import MathL3Encoder
    enc = MathL3Encoder()
    # expand: degree stays the same (x+1)^2 -> x^2 + 2x + 1
    obs = ((x + 1)**2, sympy.expand((x + 1)**2))
    vec = enc.encode(obs, epistemic=None)[0]
    # index 6 = deg_unchanged
    assert vec[6] == 1.0


def test_l3_integrate_increases_degree():
    from hpm.encoders.math_encoders import MathL3Encoder
    enc = MathL3Encoder()
    # integrate(x^2) = x^3/3: degree 2 -> 3
    obs = (x**2, sympy.integrate(x**2, x))
    vec = enc.encode(obs, epistemic=None)[0]
    # index 4 = deg_increased
    assert vec[4] == 1.0


def test_all_encoders_on_all_families():
    """Integration: all three encoders handle all 5 family transformations."""
    from hpm.encoders.math_encoders import MathL1Encoder, MathL2Encoder, MathL3Encoder
    l1 = MathL1Encoder()
    l2 = MathL2Encoder()
    l3 = MathL3Encoder()

    pairs = [
        (x**2 + 2*x + 1, sympy.expand((x + 1)**2)),         # expand
        (x**2 - 1, sympy.factor(x**2 - 1)),                   # factor
        (x**2 - x + x, sympy.simplify(x**2 - x + x)),         # simplify
        (x**3, sympy.diff(x**3, x)),                           # differentiate
        (x**2, sympy.integrate(x**2, x)),                      # integrate
    ]

    for pair in pairs:
        v1 = l1.encode(pair, epistemic=None)
        assert len(v1) == 1 and v1[0].shape == (14,)
        v2 = l2.encode(pair, epistemic=None)
        assert len(v2) == 1 and v2[0].shape == (10,)
        v3 = l3.encode(pair, epistemic=None)
        assert len(v3) == 1 and v3[0].shape == (12,)


def test_benchmark_full_run():
    from benchmarks.structured_math import generate_tasks, run_benchmark
    tasks = generate_tasks(n_per_family=5, seed=7)
    acc = run_benchmark(tasks, 'full')
    assert 0.0 <= acc <= 1.0
