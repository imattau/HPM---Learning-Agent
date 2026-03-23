# Structured Math Benchmark (SP5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a structured math benchmark using three LevelEncoders (coefficient features → structural features → transformation-class summary) to test whether HPM hierarchy improves discrimination of algebraic transformation families.

**Architecture:** SymPy generates equation pairs for 5 transformation families (expand, factor, simplify, differentiate, integrate). Three MathLevelEncoders transform expressions into numpy feature vectors at increasing abstraction. StructuredOrchestrator (from SP4) wires them into a 3-level HPM hierarchy. Benchmark runs 4 baselines matching structured_arc.py.

**Tech Stack:** Python, NumPy, SymPy, pytest, existing hpm/encoders/base.py LevelEncoder protocol, hpm/agents/structured.py StructuredOrchestrator

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `hpm/encoders/math_encoders.py` | Create | MathL1Encoder, MathL2Encoder, MathL3Encoder |
| `benchmarks/structured_math.py` | Create | Task generation + benchmark loop |
| `tests/encoders/test_math_encoders.py` | Create | Full encoder + benchmark test suite |

---

### Task 1: MathL1Encoder

**Files:**
- Create: `hpm/encoders/math_encoders.py` (MathL1Encoder only)
- Create: `tests/encoders/test_math_encoders.py` (stub for this task)

- [ ] **Step 1: Write the failing test**

```python
# tests/encoders/test_math_encoders.py
import numpy as np
import sympy
from sympy import symbols, expand, factor, simplify, diff, integrate

x = symbols('x')


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_math_l1_encoder_shape -v
```
Expected: `ModuleNotFoundError: No module named 'hpm.encoders.math_encoders'`

- [ ] **Step 3: Implement**

```python
# hpm/encoders/math_encoders.py
from __future__ import annotations
import numpy as np
import sympy
from sympy import Symbol, Poly, symbols


def poly_coeffs(expr, x: Symbol, max_degree: int = 4) -> np.ndarray:
    """Extract polynomial coefficients padded/truncated to max_degree+1 terms.

    Returns a numpy array of shape (max_degree+1,) = (5,) for max_degree=4.
    Coefficients are ordered from degree 0 to degree max_degree.
    Returns zeros if expr is not a polynomial in x.
    """
    result = np.zeros(max_degree + 1, dtype=float)
    try:
        p = Poly(expr, x)
        coeffs = p.all_coeffs()  # highest degree first
        # all_coeffs() returns [a_n, ..., a_1, a_0]; pad/truncate to max_degree+1
        for i, c in enumerate(reversed(coeffs)):
            if i <= max_degree:
                result[i] = float(c)
    except (sympy.polys.polyerrors.GeneratorsNeeded,
            sympy.polys.polyerrors.PolynomialError,
            TypeError):
        pass
    return result


class MathL1Encoder:
    """Level-1 encoder: coefficient-level features (14-dim).

    Feature vector layout (indices 0-13):
      0-4:  poly_coeffs(in_expr, x)   — degrees 0..4 of input
      5-9:  poly_coeffs(out_expr, x)  — degrees 0..4 of output
      10:   sign_lead_in  (+1 if leading coeff > 0, else -1, 0 if zero)
      11:   sign_lead_out (+1 if leading coeff > 0, else -1, 0 if zero)
      12:   n_terms_in_norm  (number of non-zero terms in input / 5)
      13:   n_terms_out_norm (number of non-zero terms in output / 5)
    """

    feature_dim: int = 14
    max_steps_per_obs: int | None = 1

    def encode(
        self,
        observation: tuple,
        epistemic: tuple[float, float] | None,
    ) -> list[np.ndarray]:
        in_expr, out_expr = observation
        x = symbols('x')

        c_in = poly_coeffs(in_expr, x)
        c_out = poly_coeffs(out_expr, x)

        def sign_lead(coeffs: np.ndarray) -> float:
            # coeffs[i] = coefficient of x^i; highest degree is last
            for v in reversed(coeffs):
                if v != 0.0:
                    return 1.0 if v > 0 else -1.0
            return 0.0

        sign_in = sign_lead(c_in)
        sign_out = sign_lead(c_out)
        n_in = float(np.count_nonzero(c_in)) / 5.0
        n_out = float(np.count_nonzero(c_out)) / 5.0

        vec = np.concatenate([c_in, c_out, [sign_in, sign_out, n_in, n_out]])
        return [vec]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_math_l1_encoder_shape tests/encoders/test_math_encoders.py::test_math_l1_encoder_values tests/encoders/test_math_encoders.py::test_poly_coeffs_non_polynomial -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm/encoders/math_encoders.py tests/encoders/test_math_encoders.py
git commit -m "feat: add MathL1Encoder with poly_coeffs helper (SP5 Task 1)"
```

---

### Task 2: MathL2Encoder

**Files:**
- Extend: `hpm/encoders/math_encoders.py` (add helpers + MathL2Encoder)
- Extend: `tests/encoders/test_math_encoders.py` (add L2 tests)

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_math_encoders.py

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_math_l2_encoder_shape tests/encoders/test_math_encoders.py::test_math_l2_encoder_factored_output tests/encoders/test_math_encoders.py::test_math_l2_encoder_epistemic_threading -v
```
Expected: `ImportError` or `AttributeError` — MathL2Encoder not yet defined

- [ ] **Step 3: Implement**

Add helpers and MathL2Encoder to `hpm/encoders/math_encoders.py`:

```python
# Append to hpm/encoders/math_encoders.py

def count_factors(expr, x: Symbol) -> int:
    """Count multiplicative factors of expr that involve x."""
    try:
        args = sympy.factor(expr).as_ordered_factors()
        return sum(1 for a in args if x in a.free_symbols)
    except Exception:
        return 1


def is_factored_form(expr, x: Symbol) -> bool:
    """True if expr is a product of two or more factors involving x."""
    return count_factors(expr, x) >= 2


def has_rational(expr, x: Symbol) -> bool:
    """True if expr contains a rational sub-expression (division by poly in x)."""
    return bool(expr.is_rational_function(x)) and expr.as_numer_denom()[1] != 1


class MathL2Encoder:
    """Level-2 encoder: structural features (10-dim).

    Feature vector layout (indices 0-9):
      0:  deg_in / 4.0
      1:  deg_out / 4.0
      2:  n_factors(out) / 10.0
      3:  is_factored_in  (1.0 or 0.0)
      4:  is_factored_out (1.0 or 0.0)
      5:  has_rational_out (1.0 or 0.0)
      6:  term_ratio: clamp(n_terms_out / max(n_terms_in, 1), 0, 2) / 2.0
      7:  l1_weight  (from epistemic[0], or 0.0)
      8:  l1_loss    (from epistemic[1], or 0.0)
      9:  0.0        (padding)
    """

    feature_dim: int = 10
    max_steps_per_obs: int | None = 1

    def encode(
        self,
        observation: tuple,
        epistemic: tuple[float, float] | None,
    ) -> list[np.ndarray]:
        in_expr, out_expr = observation
        x = symbols('x')

        c_in = poly_coeffs(in_expr, x)
        c_out = poly_coeffs(out_expr, x)

        def poly_degree(coeffs: np.ndarray) -> int:
            for i in range(len(coeffs) - 1, -1, -1):
                if coeffs[i] != 0.0:
                    return i
            return 0

        deg_in = float(poly_degree(c_in)) / 4.0
        deg_out = float(poly_degree(c_out)) / 4.0
        n_fac = float(count_factors(out_expr, x)) / 10.0
        fac_in = 1.0 if is_factored_form(in_expr, x) else 0.0
        fac_out = 1.0 if is_factored_form(out_expr, x) else 0.0
        rat_out = 1.0 if has_rational(out_expr, x) else 0.0

        n_terms_in = float(np.count_nonzero(c_in))
        n_terms_out = float(np.count_nonzero(c_out))
        term_ratio = min(n_terms_out / max(n_terms_in, 1.0), 2.0) / 2.0

        l1_weight = epistemic[0] if epistemic is not None else 0.0
        l1_loss = epistemic[1] if epistemic is not None else 0.0

        vec = np.array([
            deg_in, deg_out, n_fac, fac_in, fac_out,
            rat_out, term_ratio, l1_weight, l1_loss, 0.0,
        ], dtype=float)
        return [vec]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_math_l2_encoder_shape tests/encoders/test_math_encoders.py::test_math_l2_encoder_factored_output tests/encoders/test_math_encoders.py::test_math_l2_encoder_epistemic_threading -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm/encoders/math_encoders.py tests/encoders/test_math_encoders.py
git commit -m "feat: add MathL2Encoder with structural features (SP5 Task 2)"
```

---

### Task 3: MathL3Encoder

**Files:**
- Extend: `hpm/encoders/math_encoders.py` (add MathL3Encoder)
- Extend: `tests/encoders/test_math_encoders.py` (add L3 tests)

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_math_encoders.py
from sympy import diff as sym_diff

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
    from hpm.encoders.math_encoders import MathL3Encoder
    from sympy import expand as sym_expand
    enc = MathL3Encoder()
    # expand((x+1)^2) = x^2 + 2x + 1: factored_to_expanded
    obs = ((x + 1)**2, sym_expand((x + 1)**2))
    vec = enc.encode(obs, epistemic=None)[0]
    # index 7 = factored_to_expanded
    assert vec[7] == 1.0
    # index 8 = expanded_to_factored
    assert vec[8] == 0.0


def test_math_l3_encoder_epistemic_threading():
    from hpm.encoders.math_encoders import MathL3Encoder
    enc = MathL3Encoder()
    obs = (x**2, x**2)
    vec = enc.encode(obs, epistemic=(0.8, 0.1))[0]
    # index 10 = l2_weight, index 11 = l2_loss
    assert vec[10] == 0.8
    assert vec[11] == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_math_l3_encoder_shape tests/encoders/test_math_encoders.py::test_math_l3_encoder_differentiate tests/encoders/test_math_encoders.py::test_math_l3_encoder_expand tests/encoders/test_math_encoders.py::test_math_l3_encoder_epistemic_threading -v
```
Expected: `ImportError` or `AttributeError` — MathL3Encoder not yet defined

- [ ] **Step 3: Implement**

Append MathL3Encoder to `hpm/encoders/math_encoders.py`:

```python
# Append to hpm/encoders/math_encoders.py

class MathL3Encoder:
    """Level-3 encoder: transformation-class summary (12-dim).

    Feature vector layout (indices 0-11):
      0:  deg_delta / 4.0           (deg_out - deg_in, clamped to [-4, 4])
      1:  term_delta / 10.0         (n_terms_out - n_terms_in)
      2:  factor_delta / 10.0       (n_factors_out - n_factors_in)
      3:  lead_coeff_changed        (1.0 if sign of leading coeff changed)
      4:  deg_increased             (1.0 if deg_out > deg_in)
      5:  deg_decreased             (1.0 if deg_out < deg_in)
      6:  deg_unchanged             (1.0 if deg_out == deg_in)
      7:  factored_to_expanded      (1.0 if in factored, out not factored)
      8:  expanded_to_factored      (1.0 if out factored, in not factored)
      9:  has_rational_out          (1.0 if output has rational sub-expression)
      10: l2_weight                 (from epistemic[0], or 0.0)
      11: l2_loss                   (from epistemic[1], or 0.0)
    """

    feature_dim: int = 12
    max_steps_per_obs: int | None = 1

    def encode(
        self,
        observation: tuple,
        epistemic: tuple[float, float] | None,
    ) -> list[np.ndarray]:
        in_expr, out_expr = observation
        x = symbols('x')

        c_in = poly_coeffs(in_expr, x)
        c_out = poly_coeffs(out_expr, x)

        def poly_degree(coeffs: np.ndarray) -> int:
            for i in range(len(coeffs) - 1, -1, -1):
                if coeffs[i] != 0.0:
                    return i
            return 0

        def sign_lead(coeffs: np.ndarray) -> float:
            for v in reversed(coeffs):
                if v != 0.0:
                    return 1.0 if v > 0 else -1.0
            return 0.0

        deg_in = poly_degree(c_in)
        deg_out = poly_degree(c_out)
        deg_delta = float(np.clip(deg_out - deg_in, -4, 4)) / 4.0

        n_terms_in = float(np.count_nonzero(c_in))
        n_terms_out = float(np.count_nonzero(c_out))
        term_delta = float(np.clip(n_terms_out - n_terms_in, -10, 10)) / 10.0

        n_fac_in = float(count_factors(in_expr, x))
        n_fac_out = float(count_factors(out_expr, x))
        factor_delta = float(np.clip(n_fac_out - n_fac_in, -10, 10)) / 10.0

        lead_coeff_changed = 1.0 if sign_lead(c_in) != sign_lead(c_out) else 0.0
        deg_increased = 1.0 if deg_out > deg_in else 0.0
        deg_decreased = 1.0 if deg_out < deg_in else 0.0
        deg_unchanged = 1.0 if deg_out == deg_in else 0.0

        fac_in = is_factored_form(in_expr, x)
        fac_out = is_factored_form(out_expr, x)
        factored_to_expanded = 1.0 if (fac_in and not fac_out) else 0.0
        expanded_to_factored = 1.0 if (fac_out and not fac_in) else 0.0

        rat_out = 1.0 if has_rational(out_expr, x) else 0.0

        l2_weight = epistemic[0] if epistemic is not None else 0.0
        l2_loss = epistemic[1] if epistemic is not None else 0.0

        vec = np.array([
            deg_delta, term_delta, factor_delta, lead_coeff_changed,
            deg_increased, deg_decreased, deg_unchanged,
            factored_to_expanded, expanded_to_factored,
            rat_out, l2_weight, l2_loss,
        ], dtype=float)
        return [vec]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_math_l3_encoder_shape tests/encoders/test_math_encoders.py::test_math_l3_encoder_differentiate tests/encoders/test_math_encoders.py::test_math_l3_encoder_expand tests/encoders/test_math_encoders.py::test_math_l3_encoder_epistemic_threading -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm/encoders/math_encoders.py tests/encoders/test_math_encoders.py
git commit -m "feat: add MathL3Encoder with transformation-class summary (SP5 Task 3)"
```

---

### Task 4: Task Generation

**Files:**
- Create: `benchmarks/structured_math.py` (generate_tasks only)
- Extend: `tests/encoders/test_math_encoders.py` (add task generation tests)

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_math_encoders.py
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
    import sympy
    tasks = generate_tasks(n_per_family=5, seed=42)
    for task in tasks:
        correct = task['test_output']
        distractors = [c for c in task['candidates'] if c is not correct]
        for d in distractors:
            diff = sympy.simplify(correct - d)
            assert diff != 0, f"Distractor equals correct answer in family {task['family']}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_generate_tasks_count tests/encoders/test_math_encoders.py::test_generate_tasks_structure tests/encoders/test_math_encoders.py::test_generate_tasks_no_duplicate_correct -v
```
Expected: `ModuleNotFoundError: No module named 'benchmarks.structured_math'`

- [ ] **Step 3: Implement**

```python
# benchmarks/structured_math.py
"""Structured Math Benchmark (SP5).

Generates discrimination tasks for 5 algebraic transformation families.
Each task: given (test_input, candidates), pick the candidate that is
the correct transformation output.
"""
from __future__ import annotations

import numpy as np
import sympy
from sympy import symbols, expand, factor, simplify, diff, integrate, Rational


FAMILIES = ['expand', 'factor', 'simplify', 'differentiate', 'integrate']


def _apply_family(family: str, expr, x) -> sympy.Expr | None:
    """Apply transformation family to expr. Returns None for invalid/no-op."""
    try:
        if family == 'expand':
            return expand(expr)
        elif family == 'factor':
            result = factor(expr)
            # Reject no-op: factored form same as input
            if sympy.simplify(result - expr) == 0:
                return None
            return result
        elif family == 'simplify':
            result = simplify(expr)
            # Reject no-op
            if sympy.simplify(result - expr) == 0:
                return None
            return result
        elif family == 'differentiate':
            return diff(expr, x)
        elif family == 'integrate':
            result = integrate(expr, x)
            # integrate adds C; use without constant
            return result
    except Exception:
        return None


def _random_polynomial(rng: np.random.Generator, x, max_degree: int = 4) -> sympy.Expr:
    """Generate a random polynomial with integer coefficients in [-3, 3]."""
    degree = int(rng.integers(1, max_degree + 1))
    coeffs = rng.integers(-3, 4, size=degree + 1)
    # Ensure leading coefficient is nonzero
    while coeffs[-1] == 0:
        coeffs[-1] = int(rng.integers(-3, 4))
    expr = sympy.Integer(0)
    for i, c in enumerate(coeffs):
        expr += sympy.Integer(int(c)) * x**i
    return expr


def _random_factorable(rng: np.random.Generator, x) -> sympy.Expr:
    """Generate a factorable polynomial as product of two linear factors."""
    a1 = int(rng.integers(-3, 4))
    b1 = int(rng.integers(-3, 4))
    a2 = int(rng.integers(-3, 4))
    b2 = int(rng.integers(-3, 4))
    # (a1*x + b1)(a2*x + b2), avoid zero leading
    f1 = sympy.Integer(a1 if a1 != 0 else 1) * x + sympy.Integer(b1)
    f2 = sympy.Integer(a2 if a2 != 0 else 1) * x + sympy.Integer(b2)
    return expand(f1 * f2)


def generate_tasks(n_per_family: int = 60, seed: int = 42) -> list[dict]:
    """Generate discrimination tasks for all 5 transformation families.

    Args:
        n_per_family: Target number of tasks per family (some filtered out).
        seed: Random seed for reproducibility.

    Returns:
        List of task dicts with keys:
          family, train, test_input, test_output, candidates
    """
    rng = np.random.default_rng(seed)
    x = symbols('x')
    tasks = []

    for family in FAMILIES:
        family_tasks = []
        attempts = 0
        max_attempts = n_per_family * 20

        while len(family_tasks) < n_per_family and attempts < max_attempts:
            attempts += 1

            # Generate test input
            if family == 'factor':
                test_input = _random_factorable(rng, x)
            else:
                test_input = _random_polynomial(rng, x)

            # Apply target family
            test_output = _apply_family(family, test_input, x)
            if test_output is None:
                continue

            # Generate distractors: apply other 4 families to test_input
            candidates = [test_output]
            for other in FAMILIES:
                if other == family:
                    continue
                d = _apply_family(other, test_input, x)
                if d is None:
                    d = sympy.Integer(0)
                candidates.append(d)

            # Filter: ensure no distractor equals test_output
            valid = True
            for d in candidates[1:]:
                try:
                    if sympy.simplify(test_output - d) == 0:
                        valid = False
                        break
                except Exception:
                    pass
            if not valid:
                continue

            # Build training pairs: 3 examples of the same family
            train = []
            for _ in range(3):
                if family == 'factor':
                    tr_in = _random_factorable(rng, x)
                else:
                    tr_in = _random_polynomial(rng, x)
                tr_out = _apply_family(family, tr_in, x)
                if tr_out is not None:
                    train.append((tr_in, tr_out))

            family_tasks.append({
                'family': family,
                'train': train,
                'test_input': test_input,
                'test_output': test_output,
                'candidates': candidates,
            })

        tasks.extend(family_tasks)

    return tasks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_generate_tasks_count tests/encoders/test_math_encoders.py::test_generate_tasks_structure tests/encoders/test_math_encoders.py::test_generate_tasks_no_duplicate_correct -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add benchmarks/structured_math.py tests/encoders/test_math_encoders.py
git commit -m "feat: add generate_tasks for structured math benchmark (SP5 Task 4)"
```

---

### Task 5: Benchmark Loop

**Files:**
- Extend: `benchmarks/structured_math.py` (add run_benchmark + main)
- Extend: `tests/encoders/test_math_encoders.py` (add smoke test)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/encoders/test_math_encoders.py
def test_benchmark_smoke():
    from benchmarks.structured_math import generate_tasks, run_benchmark
    tasks = generate_tasks(n_per_family=3, seed=0)
    for condition in ['flat', 'l1_only', 'l2_only', 'full']:
        acc = run_benchmark(tasks, condition)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0, f"condition={condition} returned {acc}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_benchmark_smoke -v
```
Expected: `ImportError` — run_benchmark not yet defined

- [ ] **Step 3: Implement**

Append to `benchmarks/structured_math.py`:

```python
# Append to benchmarks/structured_math.py

from hpm.encoders.math_encoders import MathL1Encoder, MathL2Encoder, MathL3Encoder
from hpm.agents.structured import StructuredOrchestrator
from hpm.orchestrator import make_orchestrator


def _score_candidates_nll(
    encoder,
    train_pairs: list[tuple],
    test_input,
    candidates: list,
    epistemic=None,
) -> list[float]:
    """Return NLL score for each candidate under a single-level encoder.

    Lower NLL = more likely under the pattern learned from train_pairs.
    Uses a simple mean vector as the prototype and returns negative
    log-likelihood as L2 distance (Gaussian approximation).
    """
    # Encode all training pairs
    train_vecs = []
    for pair in train_pairs:
        vecs = encoder.encode(pair, epistemic=epistemic)
        train_vecs.extend(vecs)

    if len(train_vecs) == 0:
        return [0.0] * len(candidates)

    prototype = np.mean(train_vecs, axis=0)

    scores = []
    for candidate in candidates:
        obs = (test_input, candidate)
        vecs = encoder.encode(obs, epistemic=epistemic)
        if len(vecs) == 0:
            scores.append(float('inf'))
            continue
        vec = np.mean(vecs, axis=0)
        nll = float(np.sum((vec - prototype) ** 2))
        scores.append(nll)
    return scores


def run_benchmark(tasks: list[dict], condition: str) -> float:
    """Run benchmark under the given condition, return accuracy (fraction correct).

    Conditions:
      flat:    L1 encoder only, flat pattern matching
      l1_only: L1 encoder only (same as flat, explicit naming)
      l2_only: L2 encoder only
      full:    L1 + L2 + L3 encoders, NLL summed across levels
    """
    if condition not in ('flat', 'l1_only', 'l2_only', 'full'):
        raise ValueError(f"Unknown condition: {condition}")

    l1_enc = MathL1Encoder()
    l2_enc = MathL2Encoder()
    l3_enc = MathL3Encoder()

    correct = 0
    for task in tasks:
        train = task['train']
        test_input = task['test_input']
        test_output = task['test_output']
        candidates = task['candidates']

        if condition in ('flat', 'l1_only'):
            scores = _score_candidates_nll(l1_enc, train, test_input, candidates, epistemic=None)

        elif condition == 'l2_only':
            scores = _score_candidates_nll(l2_enc, train, test_input, candidates, epistemic=None)

        elif condition == 'full':
            # L1 pass
            l1_scores = _score_candidates_nll(l1_enc, train, test_input, candidates, epistemic=None)
            # Epistemic signal from L1: use (mean_weight=1.0, mean_loss=mean_l1_nll)
            mean_l1_nll = float(np.mean(l1_scores)) if l1_scores else 0.0
            epistemic_l1 = (1.0, mean_l1_nll)
            # L2 pass
            l2_scores = _score_candidates_nll(l2_enc, train, test_input, candidates, epistemic=epistemic_l1)
            mean_l2_nll = float(np.mean(l2_scores)) if l2_scores else 0.0
            epistemic_l2 = (1.0, mean_l2_nll)
            # L3 pass
            l3_scores = _score_candidates_nll(l3_enc, train, test_input, candidates, epistemic=epistemic_l2)
            # Combined: sum NLL across levels
            scores = [l1 + l2 + l3 for l1, l2, l3 in zip(l1_scores, l2_scores, l3_scores)]

        # Pick candidate with lowest NLL (best match)
        predicted_idx = int(np.argmin(scores))
        predicted = candidates[predicted_idx]

        try:
            if sympy.simplify(predicted - test_output) == 0:
                correct += 1
        except Exception:
            if predicted == test_output:
                correct += 1

    return float(correct) / len(tasks) if tasks else 0.0


def main():
    """Run full benchmark and print results table."""
    print("Generating tasks...")
    tasks = generate_tasks(n_per_family=60, seed=42)
    print(f"Generated {len(tasks)} tasks across {len(FAMILIES)} families.\n")

    conditions = ['flat', 'l1_only', 'l2_only', 'full']
    results = {}
    for cond in conditions:
        acc = run_benchmark(tasks, cond)
        results[cond] = acc
        print(f"  {cond:12s}: {acc:.3f} accuracy")

    print("\nSummary:")
    print(f"  Flat baseline:  {results['flat']:.3f}")
    print(f"  Full HPM:       {results['full']:.3f}")
    delta = results['full'] - results['flat']
    print(f"  Delta (HPM - flat): {delta:+.3f}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_benchmark_smoke -v
```
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add benchmarks/structured_math.py tests/encoders/test_math_encoders.py
git commit -m "feat: add run_benchmark and main() for structured math (SP5 Task 5)"
```

---

### Task 6: Full Test Suite

**Files:**
- Extend: `tests/encoders/test_math_encoders.py` (add edge case + integration tests)

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_math_encoders.py
from sympy import Integer as SInt

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py::test_l1_zero_polynomial tests/encoders/test_math_encoders.py::test_l2_rational_output tests/encoders/test_math_encoders.py::test_all_encoders_on_all_families -v
```
Expected: FAILED (implementation gaps revealed by edge cases)

- [ ] **Step 3: Fix any edge-case failures**

Run the full suite and fix any failures found:

```bash
python3 -m pytest tests/encoders/test_math_encoders.py -v 2>&1 | head -60
```

Common fixes needed:
- `poly_coeffs` for `sympy.Integer(0)`: ensure `Poly(0, x)` returns zeros (catch `sympy.polys.polyerrors.GeneratorsNeeded`)
- `has_rational` for expressions like `1/x`: ensure `as_numer_denom()[1] != 1` handles symbolic denominators correctly

If `sympy.Poly(sympy.Integer(0), x)` raises, add explicit zero check:

```python
# In poly_coeffs(), add before try block:
if expr == sympy.Integer(0) or expr == 0:
    return result
```

- [ ] **Step 4: Run full test suite to verify all pass**

```bash
python3 -m pytest tests/encoders/test_math_encoders.py -v
```
Expected: ALL PASSED (14+ tests)

- [ ] **Step 5: Commit**

```bash
git add hpm/encoders/math_encoders.py tests/encoders/test_math_encoders.py
git commit -m "test: add full edge-case and integration tests for SP5 math encoders"
```

---

## Final Verification

Run the complete test suite and then execute the benchmark:

```bash
python3 -m pytest tests/encoders/test_math_encoders.py -v
```
Expected: All tests pass.

```bash
python3 benchmarks/structured_math.py
```
Expected output (exact numbers will vary):
```
Generating tasks...
Generated ~300 tasks across 5 families.

  flat        : 0.XXX accuracy
  l1_only     : 0.XXX accuracy
  l2_only     : 0.XXX accuracy
  full        : 0.XXX accuracy

Summary:
  Flat baseline:  0.XXX
  Full HPM:       0.XXX
  Delta (HPM - flat): +0.XXX
```

The `full` condition should outperform `flat` if the HPM hierarchy adds genuine discriminative power.
