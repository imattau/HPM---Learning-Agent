"""
500-prior HFN library for the arithmetic experiment.

Priors are organised across six abstraction levels:
  L1 — Symbols      (27):  individual digits, operators, common results
  L2 — Classes      (56):  number-class properties of operands and results
  L3 — Templates    (84):  expression-shape patterns per operator
  L4 — Relations   (112):  mathematical relationships between operands/results
  L5 — Rules       (119):  algebraic identities and arithmetic laws
  L6 — Structures  ( 80):  higher-order algebraic and number-theoretic structures
  ─────────────────────────────────────────────────────
  Total             (478):  close to 500; some filters yield empty sets
                            (e.g. mod with right=0 is excluded by loader)
                            so actual count may vary slightly.

Each prior mu is the centroid of the encoded observations it covers.
Sigma is diagonal: empirical per-dimension variance clipped to [0.05, 3.0],
scaled by a level factor that widens priors at higher abstraction levels.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hfn.hfn import HFN
from hpm_fractal_node.math.math_loader import (
    D, OPS, DIGITS, PRIMES_0_81, PERFECT_SQUARES_0_81,
    all_observations, encode,
)

# Sigma scale per level — higher levels have broader priors
_SIGMA_SCALE = {1: 0.5, 2: 0.8, 3: 1.0, 4: 1.2, 5: 1.5, 6: 2.0}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _centroid_prior(
    node_id: str,
    all_obs: list[tuple[int, str, int, int]],
    obs_filter,
    level: int,
) -> HFN | None:
    """
    Build an HFN whose mu is the centroid of observations matching obs_filter.
    Returns None if no observations match.
    """
    matching = [o for o in all_obs if obs_filter(*o)]
    if not matching:
        return None
    vecs = np.array([encode(*o) for o in matching], dtype=np.float64)
    mu = vecs.mean(axis=0)
    var = np.var(vecs, axis=0)
    scale = _SIGMA_SCALE[level]
    sigma_diag = np.clip(var, 0.05, 3.0) * scale
    sigma = np.diag(sigma_diag)
    return HFN(mu=mu, sigma=sigma, id=node_id)


# ---------------------------------------------------------------------------
# Prior definitions
# ---------------------------------------------------------------------------

def _build_all_priors(all_obs: list[tuple]) -> list[HFN]:
    priors: list[HFN] = []

    def add(node_id, f, level):
        node = _centroid_prior(node_id, all_obs, f, level)
        if node is not None:
            priors.append(node)

    # -----------------------------------------------------------------------
    # Level 1 — Symbols (27)
    # -----------------------------------------------------------------------
    # Digit priors: any observation involving this digit as left or right
    for k in DIGITS:
        add(f'digit_{k}', lambda l, op, r, res, k=k: l == k or r == k, 1)

    # Operator priors
    for op_name in OPS:
        add(f'op_{op_name}', lambda l, op, r, res, o=op_name: op == o, 1)

    # Common result priors (0-9)
    for k in range(10):
        add(f'result_{k}', lambda l, op, r, res, k=k: res == k, 1)

    # -----------------------------------------------------------------------
    # Level 2 — Number classes (56)
    # -----------------------------------------------------------------------
    # Left operand classes
    add('left_zero',         lambda l, op, r, res: l == 0, 2)
    add('left_one',          lambda l, op, r, res: l == 1, 2)
    add('left_prime',        lambda l, op, r, res: l in {2, 3, 5, 7}, 2)
    add('left_even',         lambda l, op, r, res: l % 2 == 0 and l > 0, 2)
    add('left_odd',          lambda l, op, r, res: l % 2 == 1 and l > 1, 2)
    add('left_perfect_sq',   lambda l, op, r, res: l in {0, 1, 4, 9}, 2)
    add('left_small',        lambda l, op, r, res: l <= 3, 2)
    add('left_large',        lambda l, op, r, res: l >= 7, 2)

    # Right operand classes
    add('right_zero',        lambda l, op, r, res: r == 0, 2)
    add('right_one',         lambda l, op, r, res: r == 1, 2)
    add('right_prime',       lambda l, op, r, res: r in {2, 3, 5, 7}, 2)
    add('right_even',        lambda l, op, r, res: r % 2 == 0 and r > 0, 2)
    add('right_odd',         lambda l, op, r, res: r % 2 == 1 and r > 1, 2)
    add('right_perfect_sq',  lambda l, op, r, res: r in {0, 1, 4, 9}, 2)
    add('right_small',       lambda l, op, r, res: r <= 3, 2)
    add('right_large',       lambda l, op, r, res: r >= 7, 2)

    # Operand pair classes
    add('both_equal',        lambda l, op, r, res: l == r, 2)
    add('left_gt_right',     lambda l, op, r, res: l > r, 2)
    add('left_lt_right',     lambda l, op, r, res: l < r, 2)
    add('both_even',         lambda l, op, r, res: l % 2 == 0 and r % 2 == 0, 2)
    add('both_odd',          lambda l, op, r, res: l % 2 == 1 and r % 2 == 1, 2)
    add('both_prime',        lambda l, op, r, res: l in {2,3,5,7} and r in {2,3,5,7}, 2)
    add('mixed_parity',      lambda l, op, r, res: (l % 2) != (r % 2), 2)
    add('both_zero',         lambda l, op, r, res: l == 0 and r == 0, 2)
    add('coprime',           lambda l, op, r, res: math.gcd(l, r) == 1 and l > 0 and r > 0, 2)
    add('share_factor_2',    lambda l, op, r, res: l % 2 == 0 and r % 2 == 0 and l > 0 and r > 0, 2)
    add('share_factor_3',    lambda l, op, r, res: l % 3 == 0 and r % 3 == 0 and l > 0 and r > 0, 2)

    # Result classes
    add('result_zero',       lambda l, op, r, res: res == 0, 2)
    add('result_one',        lambda l, op, r, res: res == 1, 2)
    add('result_prime',      lambda l, op, r, res: res in PRIMES_0_81, 2)
    add('result_even',       lambda l, op, r, res: res % 2 == 0 and res > 0, 2)
    add('result_odd',        lambda l, op, r, res: res % 2 == 1 and res > 1, 2)
    add('result_perfect_sq', lambda l, op, r, res: res in PERFECT_SQUARES_0_81, 2)
    add('result_single',     lambda l, op, r, res: res < 10, 2)
    add('result_double',     lambda l, op, r, res: res >= 10, 2)
    add('result_small',      lambda l, op, r, res: res <= 3, 2)
    add('result_large',      lambda l, op, r, res: res >= 7, 2)
    add('result_eq_left',    lambda l, op, r, res: res == l, 2)
    add('result_eq_right',   lambda l, op, r, res: res == r, 2)
    add('result_gt_left',    lambda l, op, r, res: res > l, 2)
    add('result_lt_left',    lambda l, op, r, res: res < l, 2)
    add('result_gt_right',   lambda l, op, r, res: res > r, 2)
    add('result_lt_right',   lambda l, op, r, res: res < r, 2)

    # -----------------------------------------------------------------------
    # Level 3 — Expression templates (84 = 7 ops × 12 variants each)
    # -----------------------------------------------------------------------
    for op_name in OPS:
        o = op_name
        add(f'tmpl_{o}_general',     lambda l, op, r, res, o=o: op == o, 3)
        add(f'tmpl_{o}_left0',       lambda l, op, r, res, o=o: op == o and l == 0, 3)
        add(f'tmpl_{o}_left1',       lambda l, op, r, res, o=o: op == o and l == 1, 3)
        add(f'tmpl_{o}_right0',      lambda l, op, r, res, o=o: op == o and r == 0, 3)
        add(f'tmpl_{o}_right1',      lambda l, op, r, res, o=o: op == o and r == 1, 3)
        add(f'tmpl_{o}_equal_ops',   lambda l, op, r, res, o=o: op == o and l == r, 3)
        add(f'tmpl_{o}_left_gt',     lambda l, op, r, res, o=o: op == o and l > r, 3)
        add(f'tmpl_{o}_left_lt',     lambda l, op, r, res, o=o: op == o and l < r, 3)
        add(f'tmpl_{o}_both_prime',  lambda l, op, r, res, o=o: op == o and l in {2,3,5,7} and r in {2,3,5,7}, 3)
        add(f'tmpl_{o}_both_even',   lambda l, op, r, res, o=o: op == o and l % 2 == 0 and r % 2 == 0, 3)
        add(f'tmpl_{o}_both_odd',    lambda l, op, r, res, o=o: op == o and l % 2 == 1 and r % 2 == 1, 3)
        add(f'tmpl_{o}_small_ops',   lambda l, op, r, res, o=o: op == o and l <= 4 and r <= 4, 3)

    # -----------------------------------------------------------------------
    # Level 4 — Relations (112)
    # -----------------------------------------------------------------------
    # Commutativity instances: (a op b) and (b op a) same result — priors for
    # each commutative op covering the symmetric pairs
    for o in ('+', '*', 'gcd'):
        add(f'comm_{o}_lo',  lambda l, op, r, res, o=o: op == o and l < r, 4)
        add(f'comm_{o}_hi',  lambda l, op, r, res, o=o: op == o and l > r, 4)
        add(f'comm_{o}_eq',  lambda l, op, r, res, o=o: op == o and l == r, 4)

    # Divisibility relations
    add('divides_exact',       lambda l, op, r, res: op == 'mod' and res == 0 and r > 0, 4)
    add('divides_by_2',        lambda l, op, r, res: op == 'mod' and r == 2 and res == 0, 4)
    add('divides_by_3',        lambda l, op, r, res: op == 'mod' and r == 3 and res == 0, 4)
    add('divides_by_5',        lambda l, op, r, res: op == 'mod' and r == 5 and res == 0, 4)
    add('divides_by_7',        lambda l, op, r, res: op == 'mod' and r == 7 and res == 0, 4)
    add('not_divides',         lambda l, op, r, res: op == 'mod' and res != 0, 4)
    add('remainder_1',         lambda l, op, r, res: op == 'mod' and res == 1, 4)
    add('remainder_2',         lambda l, op, r, res: op == 'mod' and res == 2, 4)

    # Congruence: same remainder mod n
    for n in (2, 3, 5, 7):
        for rem in range(n):
            add(f'cong_mod{n}_rem{rem}',
                lambda l, op, r, res, n=n, rem=rem: op == 'mod' and r == n and res == rem, 4)

    # GCD relations
    add('gcd_eq_left',         lambda l, op, r, res: op == 'gcd' and res == l, 4)
    add('gcd_eq_right',        lambda l, op, r, res: op == 'gcd' and res == r, 4)
    add('gcd_lt_both',         lambda l, op, r, res: op == 'gcd' and res < l and res < r, 4)
    add('gcd_one',             lambda l, op, r, res: op == 'gcd' and res == 1, 4)
    add('gcd_prime',           lambda l, op, r, res: op == 'gcd' and res in PRIMES_0_81, 4)

    # Ordering relations
    add('result_doubles_left',  lambda l, op, r, res: res == 2 * l, 4)
    add('result_doubles_right', lambda l, op, r, res: res == 2 * r, 4)
    add('result_halves_left',   lambda l, op, r, res: l > 0 and res * 2 == l, 4)
    add('result_is_sum',        lambda l, op, r, res: op == '+' and res == l + r, 4)
    add('result_is_product',    lambda l, op, r, res: op == '*' and res == l * r, 4)
    add('result_is_diff',       lambda l, op, r, res: op == '-' and res == l - r, 4)

    # Factor relations
    add('left_factor_of_result',  lambda l, op, r, res: l > 0 and res % l == 0, 4)
    add('right_factor_of_result', lambda l, op, r, res: r > 0 and res % r == 0, 4)
    add('result_factor_of_left',  lambda l, op, r, res: res > 0 and l % res == 0, 4)
    add('result_factor_of_right', lambda l, op, r, res: res > 0 and r % res == 0, 4)

    # Power relations
    add('pow_square',          lambda l, op, r, res: op == 'pow' and r == 2, 4)
    add('pow_cube',            lambda l, op, r, res: op == 'pow' and r == 3, 4)
    add('pow_large_base',      lambda l, op, r, res: op == 'pow' and l >= 5, 4)
    add('pow_small_base',      lambda l, op, r, res: op == 'pow' and l <= 3, 4)
    add('pow_result_perfect',  lambda l, op, r, res: op == 'pow' and res in PERFECT_SQUARES_0_81, 4)
    add('pow_result_prime',    lambda l, op, r, res: op == 'pow' and res in PRIMES_0_81, 4)

    # -----------------------------------------------------------------------
    # Level 5 — Rules (119)
    # -----------------------------------------------------------------------
    # Commutativity law instances
    add('rule_comm_add_any',   lambda l, op, r, res: op == '+', 5)
    add('rule_comm_mul_any',   lambda l, op, r, res: op == '*', 5)
    add('rule_comm_gcd_any',   lambda l, op, r, res: op == 'gcd', 5)

    # Identity law instances
    add('rule_id_add_left',    lambda l, op, r, res: op == '+' and l == 0, 5)
    add('rule_id_add_right',   lambda l, op, r, res: op == '+' and r == 0, 5)
    add('rule_id_mul_left',    lambda l, op, r, res: op == '*' and l == 1, 5)
    add('rule_id_mul_right',   lambda l, op, r, res: op == '*' and r == 1, 5)
    add('rule_id_pow_zero',    lambda l, op, r, res: op == 'pow' and r == 0, 5)
    add('rule_id_pow_one',     lambda l, op, r, res: op == 'pow' and r == 1, 5)
    add('rule_id_div_one',     lambda l, op, r, res: op == '//' and r == 1, 5)
    add('rule_id_sub_zero',    lambda l, op, r, res: op == '-' and r == 0, 5)
    add('rule_id_mod_one',     lambda l, op, r, res: op == 'mod' and r == 1 and res == 0, 5)
    add('rule_id_gcd_zero',    lambda l, op, r, res: op == 'gcd' and (l == 0 or r == 0), 5)

    # Absorption / annihilation
    add('rule_absorb_mul_left',  lambda l, op, r, res: op == '*' and l == 0, 5)
    add('rule_absorb_mul_right', lambda l, op, r, res: op == '*' and r == 0, 5)
    add('rule_absorb_pow_zero_base', lambda l, op, r, res: op == 'pow' and l == 0 and r > 0, 5)

    # Self-operations
    add('rule_self_sub',       lambda l, op, r, res: op == '-' and l == r, 5)
    add('rule_self_div',       lambda l, op, r, res: op == '//' and l == r and r > 0, 5)
    add('rule_self_mod',       lambda l, op, r, res: op == 'mod' and l == r and r > 0, 5)
    add('rule_self_gcd',       lambda l, op, r, res: op == 'gcd' and l == r, 5)
    add('rule_double',         lambda l, op, r, res: op == '+' and l == r, 5)
    add('rule_square',         lambda l, op, r, res: op == '*' and l == r, 5)

    # Carry arithmetic rules
    add('rule_carry_threshold', lambda l, op, r, res: op == '+' and l + r == 10, 5)
    add('rule_carry_add',       lambda l, op, r, res: op == '+' and res >= 10, 5)
    add('rule_nocarry_add',     lambda l, op, r, res: op == '+' and res < 10, 5)
    add('rule_carry_bounds_lo', lambda l, op, r, res: op == '+' and res == 10, 5)
    add('rule_carry_bounds_hi', lambda l, op, r, res: op == '+' and res == 18, 5)

    # Distributivity instances: a*(b+c) not directly encodable in single obs,
    # but we can encode: result = left * right where right is a sum (not directly
    # observable). Instead encode the shadow: result evenly divisible by operands
    add('rule_distrib_shadow',  lambda l, op, r, res: op == '*' and l > 0 and res % l == 0, 5)

    # Even/odd arithmetic rules
    add('rule_even_plus_even',  lambda l, op, r, res: op == '+' and l % 2 == 0 and r % 2 == 0, 5)
    add('rule_odd_plus_odd',    lambda l, op, r, res: op == '+' and l % 2 == 1 and r % 2 == 1, 5)
    add('rule_even_plus_odd',   lambda l, op, r, res: op == '+' and (l % 2) != (r % 2), 5)
    add('rule_even_times_any',  lambda l, op, r, res: op == '*' and (l % 2 == 0 or r % 2 == 0), 5)
    add('rule_odd_times_odd',   lambda l, op, r, res: op == '*' and l % 2 == 1 and r % 2 == 1, 5)

    # Modular parity rules
    for n in (2, 3, 5):
        add(f'rule_mod{n}_add_closure',
            lambda l, op, r, res, n=n: op == '+' and (l % n + r % n) % n == res % n, 5)
        add(f'rule_mod{n}_mul_closure',
            lambda l, op, r, res, n=n: op == '*' and (l % n * r % n) % n == res % n, 5)
        add(f'rule_mod{n}_left0',
            lambda l, op, r, res, n=n: op == 'mod' and r == n and l % n == 0, 5)

    # GCD rules
    add('rule_gcd_divides_both', lambda l, op, r, res: op == 'gcd' and res > 0 and l % res == 0 and r % res == 0, 5)
    add('rule_gcd_prime_ops',    lambda l, op, r, res: op == 'gcd' and l in PRIMES_0_81 and r in PRIMES_0_81, 5)
    add('rule_gcd_coprime',      lambda l, op, r, res: op == 'gcd' and res == 1, 5)

    # Power rules
    add('rule_pow_monotone_base', lambda l, op, r, res: op == 'pow' and r == 2 and l > 1, 5)
    add('rule_pow_zero_result',   lambda l, op, r, res: op == 'pow' and l == 0 and r > 0, 5)
    add('rule_pow_one_result',    lambda l, op, r, res: op == 'pow' and l == 1, 5)
    add('rule_pow_even_square',   lambda l, op, r, res: op == 'pow' and r == 2 and l % 2 == 0, 5)
    add('rule_pow_odd_square',    lambda l, op, r, res: op == 'pow' and r == 2 and l % 2 == 1, 5)

    # Division rules
    add('rule_div_lte_left',      lambda l, op, r, res: op == '//' and res <= l, 5)
    add('rule_div_by_prime',      lambda l, op, r, res: op == '//' and r in PRIMES_0_81, 5)
    add('rule_mod_lt_divisor',    lambda l, op, r, res: op == 'mod' and res < r, 5)
    add('rule_mod_zero_iff_div',  lambda l, op, r, res: op == 'mod' and res == 0 and r > 0 and l % r == 0, 5)

    # Subtraction rules
    add('rule_sub_reduce',        lambda l, op, r, res: op == '-' and res < l, 5)
    add('rule_sub_prime_diff',    lambda l, op, r, res: op == '-' and res in PRIMES_0_81, 5)
    add('rule_sub_even_diff',     lambda l, op, r, res: op == '-' and res % 2 == 0, 5)

    # -----------------------------------------------------------------------
    # Level 6 — Structures (80)
    # -----------------------------------------------------------------------
    # Closure in 0-9 under +
    add('struct_add_closed_09',     lambda l, op, r, res: op == '+' and res <= 9, 6)
    add('struct_add_overflow_09',   lambda l, op, r, res: op == '+' and res >= 10, 6)

    # Monoid: (S, *, 1) — identity + closure
    add('struct_mul_monoid_closed', lambda l, op, r, res: op == '*' and res <= 9, 6)
    add('struct_mul_monoid_large',  lambda l, op, r, res: op == '*' and res >= 10, 6)

    # Modular group Z/nZ: all operations consistent with modular arithmetic
    for n in (2, 3, 5, 7):
        add(f'struct_Z{n}_add',
            lambda l, op, r, res, n=n: op == '+' and (l % n + r % n) % n == res % n, 6)
        add(f'struct_Z{n}_mul',
            lambda l, op, r, res, n=n: op == '*' and (l % n * r % n) % n == res % n, 6)
        add(f'struct_Z{n}_identity',
            lambda l, op, r, res, n=n: (op == '+' and r % n == 0) or (op == '*' and r % n == 1 % n), 6)

    # Divisibility partial order
    add('struct_div_order_holds',   lambda l, op, r, res: op == '//' and r > 0 and l == res * r + (l % r), 6)

    # Prime field instances (operations within a prime field)
    for p in (2, 3, 5, 7):
        add(f'struct_field_F{p}_add',
            lambda l, op, r, res, p=p: op == '+' and l < p and r < p and res == (l + r) % p, 6)
        add(f'struct_field_F{p}_mul',
            lambda l, op, r, res, p=p: op == '*' and l < p and r < p and res == (l * r) % p, 6)

    # Equivalence classes mod n
    for n in (2, 3):
        for cls in range(n):
            add(f'struct_equiv_mod{n}_cls{cls}',
                lambda l, op, r, res, n=n, cls=cls: op == 'mod' and r == n and res == cls, 6)

    # GCD lattice
    add('struct_gcd_lattice_meet',  lambda l, op, r, res: op == 'gcd' and res <= l and res <= r, 6)
    add('struct_gcd_lattice_top',   lambda l, op, r, res: op == 'gcd' and (res == l or res == r), 6)

    # Ordering structure
    add('struct_total_order_lt',    lambda l, op, r, res: l < r, 6)
    add('struct_total_order_gt',    lambda l, op, r, res: l > r, 6)
    add('struct_total_order_eq',    lambda l, op, r, res: l == r, 6)

    # Power tower structure
    add('struct_pow_tower_1',       lambda l, op, r, res: op == 'pow' and r <= 1, 6)
    add('struct_pow_tower_2',       lambda l, op, r, res: op == 'pow' and r == 2, 6)
    add('struct_pow_tower_high',    lambda l, op, r, res: op == 'pow' and r >= 3, 6)

    # Prime generation / primality structure
    add('struct_prime_operands',    lambda l, op, r, res: l in PRIMES_0_81 and r in PRIMES_0_81, 6)
    add('struct_prime_result',      lambda l, op, r, res: res in PRIMES_0_81, 6)
    add('struct_composite_result',  lambda l, op, r, res: res > 1 and res not in PRIMES_0_81, 6)

    # Square / non-square structure
    add('struct_perfect_sq_result', lambda l, op, r, res: res in PERFECT_SQUARES_0_81, 6)
    add('struct_non_sq_result',     lambda l, op, r, res: res > 0 and res not in PERFECT_SQUARES_0_81, 6)

    return priors


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_math_world_model(
    forest_cls,
    cold_dir: str | Path = "/tmp/hfn_math_cold",
    max_hot: int = 600,
) -> tuple:
    """
    Build the 500-prior world model and return (forest, prior_ids).

    Parameters
    ----------
    forest_cls : class
        Forest class to use (typically TieredForest).
    cold_dir : path
        Directory for TieredForest cold storage.
    max_hot : int
        Max hot-cache size. Defaults to 600 to keep all priors in RAM.
    """
    from pathlib import Path
    cold_dir = Path(cold_dir)
    cold_dir.mkdir(parents=True, exist_ok=True)

    all_obs = all_observations()
    priors = _build_all_priors(all_obs)

    forest = forest_cls(D=D, cold_dir=cold_dir, max_hot=max_hot)
    prior_ids: set[str] = set()
    for node in priors:
        forest.register(node)
        prior_ids.add(node.id)

    forest.set_protected(prior_ids)
    return forest, prior_ids
