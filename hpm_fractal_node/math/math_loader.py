"""
Math observation loader for HFN arithmetic experiment.

Observations are (left, op, right, result) tuples encoded as concatenated
one-hot vectors in R^109:

  dims  0-9:   left operand,  one-hot over 0-9
  dims 10-16:  operator,      one-hot over 7 ops
  dims 17-26:  right operand, one-hot over 0-9
  dims 27-108: result,        one-hot over 0-81

Operations: +, -, *, //, mod, gcd, pow
  - subtraction: only generated when left >= right (non-negative results)
  - division: right != 0; result = left // right
  - mod: right != 0
  - pow: filtered to result <= 81

No semantic flags. All structure must be discovered or pre-encoded as prior geometry.
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

D_LEFT = 10
D_OP = 7
D_RIGHT = 10
D_RESULT = 82       # results 0..81 inclusive
D = D_LEFT + D_OP + D_RIGHT + D_RESULT   # 109

OPS = ('+', '-', '*', '//', 'mod', 'gcd', 'pow')
OP_INDEX: dict[str, int] = {op: i for i, op in enumerate(OPS)}
DIGITS = tuple(range(10))

PRIMES_0_81 = frozenset(
    n for n in range(2, 82)
    if all(n % k != 0 for k in range(2, int(n**0.5) + 1))
)
PERFECT_SQUARES_0_81 = frozenset(k * k for k in range(10))  # 0,1,4,9,16,25,36,49,64,81


# ---------------------------------------------------------------------------
# Core encoding
# ---------------------------------------------------------------------------

def encode(left: int, op: str, right: int, result: int) -> np.ndarray:
    """Encode (left, op, right, result) as a sparse one-hot vector in R^109."""
    x = np.zeros(D, dtype=np.float64)
    x[left] = 1.0
    x[D_LEFT + OP_INDEX[op]] = 1.0
    x[D_LEFT + D_OP + right] = 1.0
    if 0 <= result <= 81:
        x[D_LEFT + D_OP + D_RIGHT + result] = 1.0
    return x


def _apply(left: int, op: str, right: int) -> int | None:
    """Evaluate operation. Returns None if invalid or result outside 0-81."""
    if op == '+':
        r = left + right
    elif op == '-':
        if left < right:
            return None
        r = left - right
    elif op == '*':
        r = left * right
    elif op == '//':
        if right == 0:
            return None
        r = left // right
    elif op == 'mod':
        if right == 0:
            return None
        r = left % right
    elif op == 'gcd':
        r = math.gcd(left, right)
    elif op == 'pow':
        r = left ** right
    else:
        return None
    return r if 0 <= r <= 81 else None


# ---------------------------------------------------------------------------
# Observation generation
# ---------------------------------------------------------------------------

def all_observations() -> list[tuple[int, str, int, int]]:
    """Return all valid (left, op, right, result) tuples."""
    obs = []
    for left, op, right in product(DIGITS, OPS, DIGITS):
        r = _apply(left, op, right)
        if r is not None:
            obs.append((left, op, right, r))
    return obs


def generate_observations(
    n: int = 5000,
    seed: int = 42,
) -> list[tuple[np.ndarray, tuple[int, str, int, int]]]:
    """
    Sample n observations from the full valid set (with replacement if n > total).
    Returns list of (encoded_vec, (left, op, right, result)).
    """
    pool = all_observations()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pool), size=n, replace=n > len(pool))
    return [(encode(*pool[i]), pool[i]) for i in indices]


# ---------------------------------------------------------------------------
# Category labels (for evaluation only — never given to the agent)
# ---------------------------------------------------------------------------

def get_category(left: int, op: str, right: int, result: int) -> str:
    """
    Assign the most specific mathematical category to an observation.
    Used only to evaluate learned node purity; never passed to Observer.
    """
    # Identity elements
    if op == '+' and (left == 0 or right == 0):
        return 'identity_add'
    if op == '*' and (left == 1 or right == 1):
        return 'identity_mul'
    if op == '*' and (left == 0 or right == 0):
        return 'absorption_mul'
    if op == 'pow' and right == 0:
        return 'identity_pow_zero'   # anything^0 = 1
    if op == 'pow' and right == 1:
        return 'identity_pow_one'    # anything^1 = itself
    if op == '//' and right == 1:
        return 'identity_div_one'

    # Divisibility
    if op == 'mod' and result == 0:
        return 'exact_divisibility'
    if op == 'gcd' and (left == right):
        return 'gcd_self'
    if op == 'gcd' and (left == 0 or right == 0):
        return 'gcd_with_zero'
    if op == 'gcd' and result in PRIMES_0_81:
        return 'gcd_prime'

    # Carry / no-carry in addition
    if op == '+' and result >= 10:
        return 'carry'
    if op == '+' and result < 10:
        return 'no_carry'

    # Subtraction
    if op == '-' and left == right:
        return 'sub_self'
    if op == '-':
        return 'subtraction'

    # Prime results
    if result in PRIMES_0_81:
        return 'prime_result'

    # Perfect power
    if op == 'pow' and result in PERFECT_SQUARES_0_81:
        return 'perfect_power'
    if op == 'pow':
        return 'power_general'

    # Large vs small products
    if op == '*' and result >= 10:
        return 'mul_large'
    if op == '*':
        return 'mul_small'

    # Division/mod general
    if op == '//':
        return 'floor_div'
    if op == 'mod':
        return 'mod_general'
    if op == 'gcd':
        return 'gcd_general'

    return 'general'


CATEGORY_NAMES: tuple[str, ...] = (
    'identity_add', 'identity_mul', 'absorption_mul',
    'identity_pow_zero', 'identity_pow_one', 'identity_div_one',
    'exact_divisibility', 'gcd_self', 'gcd_with_zero', 'gcd_prime',
    'carry', 'no_carry',
    'sub_self', 'subtraction',
    'prime_result', 'perfect_power', 'power_general',
    'mul_large', 'mul_small',
    'floor_div', 'mod_general', 'gcd_general',
    'general',
)
