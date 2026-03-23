"""Math encoders for the Structured Math Benchmark (SP5).

Three LevelEncoders at increasing abstraction:
  MathL1Encoder (14-dim): coefficient-level features
  MathL2Encoder (10-dim): structural features
  MathL3Encoder (12-dim): transformation-class summary
"""
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
    # Explicit zero check
    if expr == sympy.Integer(0) or expr == 0:
        return result
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
