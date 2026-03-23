# Sub-project 5: Structured Math Benchmark — Equation Transformation Families

## Goal

Extend the three-level structured encoding pipeline (built in SP4 for ARC) to a symbolic mathematics domain. Implement three `LevelEncoder` classes for polynomial equation transformations and a new benchmark (`structured_math.py`) that validates multi-level discrimination across five algebraic transformation families. Tasks are generated on-the-fly via SymPy, eliminating any static dataset dependency.

## Architecture

### Math-Specific Encoders (`hpm/encoders/math_encoders.py`)

All three encoders implement the `LevelEncoder` protocol from `hpm/encoders/base.py`:

```python
class LevelEncoder(Protocol):
    feature_dim: int
    max_steps_per_obs: int | None
    def encode(self, observation, epistemic: tuple[float, float] | None) -> list[np.ndarray]:
        ...
```

**Observation format** for all three encoders: `(input_eq, output_eq)` where both are SymPy expressions (or strings that are parsed via `sympy.sympify`).

---

#### MathL1Encoder (`feature_dim=14`, `max_steps_per_obs=1`)

Encodes surface-level polynomial structure: coefficients and term counts. No structural reasoning.

Feature vector (14 dimensions):

```
[c0_in, c1_in, c2_in, c3_in, c4_in,   # input polynomial coefficients, degrees 0–4
                                         # extracted via Poly(expr, x).all_coeffs() padded/truncated to 5
                                         # normalised: divided by max(|coeff|, 1)
 c0_out, c1_out, c2_out, c3_out, c4_out, # output polynomial coefficients, same encoding
 sign_lead_in,                           # sign of leading coefficient of input: +1 or -1
 sign_lead_out,                          # sign of leading coefficient of output: +1 or -1
 n_terms_in / MAX_TERMS,                 # number of terms in input, normalised (MAX_TERMS=10)
 n_terms_out / MAX_TERMS]                # number of terms in output, normalised
```

Notes:
- `Poly(expr, x).all_coeffs()` returns coefficients highest-degree first. Pad with zeros on the right (lower degrees) to length 5; truncate at degree 4 if higher.
- If expression is not a polynomial in `x` (e.g. after integration), extract the polynomial part via `sympy.Poly(sympy.cancel(expr).as_numer_denom()[0], x)`.
- `n_terms`: `len(expr.as_ordered_terms())`.
- Returns list of length 1.

---

#### MathL2Encoder (`feature_dim=10`, `max_steps_per_obs=1`)

Encodes structural properties: factorisation state, degree, rational terms. One vector per `(input_eq, output_eq)` pair (fixed length — math pairs do not decompose into sub-objects like ARC grids).

Feature vector (10 dimensions):

```
[deg_in / MAX_DEG,                       # degree of input polynomial (MAX_DEG=6)
 deg_out / MAX_DEG,                      # degree of output polynomial
 n_factors / MAX_FACTORS,                # number of irreducible factors from sympy.factor_list(input_eq)
                                         # factor_list returns (content, [(factor, exp), ...])
                                         # n_factors = len(factor_list(input_eq)[1])
                                         # (MAX_FACTORS=5)
 is_factored_in,                         # 1.0 if input is a product of irreducibles (n_factors > 1 or input already irreducible)
 is_factored_out,                        # 1.0 if output is a product of irreducibles
 has_rational_out,                       # 1.0 if sympy.denom(output_eq) != 1 (output has non-trivial denominator)
 term_ratio,                             # n_terms_out / n_terms_in, clamped to [0, 3]
 l1_weight,                              # epistemic thread from L1: mean pattern weight
 l1_loss,                                # epistemic thread from L1: epistemic_loss
 0.0]                                    # reserved (padding to even 10)
```

Notes:
- Degree: `sympy.Poly(expr, x).degree()` if polynomial; else 0.
- `is_factored_in`: set to 1.0 if `sympy.factor(input_eq) != sympy.expand(input_eq)` (i.e. factoring changes the form), OR if the expression is already a single irreducible factor. Concretely: `1.0 if sympy.factor_list(input_eq)[1] and sympy.Poly(input_eq, x).degree() > 1 else 0.0`.
- `is_factored_out`: same logic applied to `output_eq`.
- `has_rational_out`: `1.0 if sympy.denom(sympy.cancel(output_eq)) != 1 else 0.0`.
- `term_ratio`: `min(n_terms_out / max(n_terms_in, 1), 3.0)`.
- Returns list of length 1.

---

#### MathL3Encoder (`feature_dim=12`, `max_steps_per_obs=1`)

Encodes relational/differential properties: what changed between input and output at the structural level. Also threads epistemic state from L2.

Feature vector (12 dimensions):

```
[deg_delta / MAX_DEG,                    # (deg_out - deg_in) / MAX_DEG, clamped [-1, 1]
 term_delta / MAX_TERMS,                 # (n_terms_out - n_terms_in) / MAX_TERMS, clamped [-1, 1]
 factor_delta / MAX_FACTORS,             # (n_factors_out - n_factors_in) / MAX_FACTORS, clamped [-1, 1]
 lead_coeff_changed,                     # 1.0 if sign of leading coefficient changed
 deg_increased,                          # 1.0 if deg_out > deg_in  (one-hot trio)
 deg_decreased,                          # 1.0 if deg_out < deg_in
 deg_unchanged,                          # 1.0 if deg_out == deg_in
 factored_to_expanded,                   # 1.0 if is_factored_in and not is_factored_out
 expanded_to_factored,                   # 1.0 if not is_factored_in and is_factored_out
 has_rational_out,                       # 1.0 if output has non-trivial denominator (same as L2)
 l2_weight,                              # epistemic thread from L2
 l2_loss]                                # epistemic thread from L2
```

Notes:
- `n_factors_in` / `n_factors_out`: `len(sympy.factor_list(expr)[1])` for input and output respectively.
- `is_factored_in` / `is_factored_out`: same definition as in L2.
- `deg_increased`, `deg_decreased`, `deg_unchanged` form a proper one-hot: exactly one is 1.0.
- Returns list of length 1.

Constants used across all encoders:
- `MAX_TERMS = 10`
- `MAX_DEG = 6`
- `MAX_FACTORS = 5`

---

### StructuredOrchestrator (reused from SP4)

`hpm/agents/structured.py` — no changes required. The math encoders slot directly into the same `encoders: list[LevelEncoder]` interface.

---

## Task Generation (`benchmarks/structured_math.py`)

### Transformation Families

Five families, generated via SymPy with `x = sympy.Symbol('x')`:

| Family | Input | Output | SymPy call |
|--------|-------|--------|------------|
| `expand` | factored form e.g. `(x+1)(x+2)` | expanded polynomial | `sympy.expand(input_eq)` |
| `factor` | expanded polynomial e.g. `x²+3x+2` | factored form | `sympy.factor(input_eq)` |
| `simplify` | expression with common factors e.g. `(x²-1)/(x-1)` | simplified | `sympy.simplify(input_eq)` |
| `differentiate` | polynomial e.g. `x³+2x` | derivative | `sympy.diff(input_eq, x)` |
| `integrate` | polynomial e.g. `3x²+2` | antiderivative (no constant) | `sympy.integrate(input_eq, x)` |

### Input Generation

For each family, inputs are randomly generated polynomials or factored forms:

- **expand / factor**: product of 2–3 linear factors `(x + a_i)` where `a_i ∈ {-5, …, 5}`, sampled without replacement. Integer roots only — ensures `factor()` is non-trivial and reverses `expand()`.
- **simplify**: rational expression `(x^2 - a^2) / (x - a)` or `((x+a)(x+b)) / (x+a)` for distinct integers `a, b`. Ensures simplification is non-trivial.
- **differentiate / integrate**: polynomial `sum(c_i * x^i for i in range(degree+1))` with integer coefficients `c_i ∈ {-4, …, 4} \ {0}` and degree in `{2, 3, 4}`.

Edge case handling:
- **factor() no-op**: if `sympy.factor(input_eq) == input_eq` (already fully factored or irreducible), regenerate. Check: `sympy.factor(expr) != expr`.
- **integrate() constant term**: integration produces a constant of integration — always omit it (use `sympy.integrate(expr, x)` which returns the antiderivative without `+C`).
- **simplify() no-op**: if `sympy.simplify(input_eq) == input_eq`, regenerate. Check: `sympy.simplify(expr) != expr`.

### Distractor Construction

**Critical design point**: all five candidates for a test item share the **same `test_input_eq`**. Distractors are produced by applying the *other four* transformation families to that same input.

```python
families = ['expand', 'factor', 'simplify', 'differentiate', 'integrate']

def apply_family(family, expr, x):
    if family == 'expand':    return sympy.expand(expr)
    if family == 'factor':    return sympy.factor(expr)
    if family == 'simplify':  return sympy.simplify(expr)
    if family == 'differentiate': return sympy.diff(expr, x)
    if family == 'integrate': return sympy.integrate(expr, x)

# For target family f and test input test_input_eq:
correct_output = apply_family(f, test_input_eq, x)
distractors = [apply_family(g, test_input_eq, x) for g in families if g != f]
# candidates = [correct_output] + distractors  (shuffled, correct index tracked)
```

If two candidates produce the same SymPy expression (e.g. `factor` and `simplify` both return the same form), the task is discarded and regenerated. Check: `sympy.simplify(a - b) == 0`.

### Task Format

Each task is a dict:
```python
{
    'family': str,             # target transformation family name
    'train': [                 # 3–4 (input_eq, output_eq) pairs as SymPy expressions
        {'input': expr, 'output': expr},
        ...
    ],
    'test_input': expr,        # SymPy expression
    'test_output': expr,       # correct output (one of the 5 candidates)
    'candidates': [expr, ...], # 5 SymPy expressions (shuffled)
    'correct_idx': int,        # index of correct candidate in candidates list
}
```

Training pairs: 3–4 per task. The test input is a fresh sample from the same family (not one of the training inputs).

### Dataset Size

~300 tasks total: 5 families × ~60 tasks each. Tasks generated fresh each run with `rng = np.random.default_rng(42)` for reproducibility.

---

## Benchmark Structure

### Baselines

Four baselines measured in a single run, mirroring `structured_arc.py`:

| Baseline | Agents | Training | Scoring |
|----------|--------|----------|---------|
| `flat` | 2, L1 features only | Partitioned L1 features | L1 score only |
| `l1_only` | 2+2+1 structured | Full structured training | L1 score only |
| `l2_only` | 2+2+1 structured | Full structured training | L2 score only |
| `full` | 2+2+1 structured | Full structured training | L1 + L2 + L3 |

`flat` uses 2 agents with partitioned training pairs (pairs_a = even-indexed, pairs_b = odd-indexed) to match multi-agent baselines from prior benchmarks. The benefit of `full` over `flat` reflects the structured hierarchy.

### Agent Configuration

| Level | Agents | feature_dim | pattern_type | Notes |
|-------|--------|-------------|--------------|-------|
| L1 | 2 | 14 | gaussian | Partitioned training pairs |
| L2 | 2 | 10 | gaussian | All agents see all pairs per step |
| L3 | 1 | 12 | gaussian | Fires every K=3 `step()` calls |

All levels: `gamma_soc=0.5`, `init_sigma=2.0`, `with_monitor=False`, `InMemoryStore`, per-task reset.

### Training Loop Per Task

```python
TRAIN_REPS = 10
pairs = task['train']   # list of {'input': expr, 'output': expr}
pairs_a = pairs[0::2] or pairs
pairs_b = pairs[1::2] or pairs
n = max(len(pairs_a), len(pairs_b))

for rep in range(TRAIN_REPS):
    for k in range(n):
        obs_a = (pairs_a[k % len(pairs_a)]['input'], pairs_a[k % len(pairs_a)]['output'])
        obs_b = (pairs_b[k % len(pairs_b)]['input'], pairs_b[k % len(pairs_b)]['output'])
        l1_obs_dict = {
            l1_agent_ids[0]: math_l1_enc.encode(obs_a, epistemic=None)[0],
            l1_agent_ids[1]: math_l1_enc.encode(obs_b, epistemic=None)[0],
        }
        structured_orch.step(obs_a, l1_obs_dict=l1_obs_dict)
        # L2/L3 always see obs_a; L3 cadence governed by StructuredOrchestrator (K=3 step() calls)
```

### Scoring At Test Time

For each of the 5 candidates `(test_input_eq, candidate_output_eq)`:

```python
# L1
feat_l1 = math_l1_enc.encode((test_input_eq, candidate_output_eq), epistemic=None)[0]
L1_score = ensemble_score(l1_agents, feat_l1)

# L2 (epistemic state from trained L1 agents)
feat_l2 = math_l2_enc.encode((test_input_eq, candidate_output_eq), epistemic=(l1_w, l1_l))[0]
L2_score = ensemble_score(l2_agents, feat_l2)

# L3 (epistemic state from trained L2 agents)
feat_l3 = math_l3_enc.encode((test_input_eq, candidate_output_eq), epistemic=(l2_w, l2_l))[0]
L3_score = ensemble_score(l3_agents, feat_l3)

total = L1_score + L2_score + L3_score  # lower = more probable = preferred
```

Epistemic state `(l1_w, l1_l)` and `(l2_w, l2_l)` are extracted from trained agents once after the training loop completes, not re-computed per candidate.

Prediction: `np.argmin([total_score(c) for c in candidates])`. Correct if `predicted_idx == task['correct_idx']`.

### Success Criteria

- `full` accuracy > `flat` accuracy
- `l1_only` vs `flat`: structured training at L1 (14-dim explicit features vs 14-dim random projection) should not regress
- At least one of `l2_only` or L3 contributes measurable accuracy above chance (>20%) — confirms each encoder level carries discriminative signal

---

## File Layout

```
hpm/encoders/math_encoders.py           — MathL1Encoder, MathL2Encoder, MathL3Encoder
benchmarks/structured_math.py          — task generation + benchmark loop (4 baselines)
tests/encoders/test_math_encoders.py    — unit tests for all three encoders
```

SymPy is added to `requirements.txt` (if not already present).

---

## Testing Approach (`tests/encoders/test_math_encoders.py`)

### MathL1Encoder

- `test_feature_dim`: `encode()` returns list of length 1, shape `(14,)`.
- `test_expand_pair`: for `(x+1)(x+2)` → `x²+3x+2`, check coefficients match expected values.
- `test_epistemic_ignored`: L1 ignores `epistemic` arg (returns same vector regardless).
- `test_coefficient_normalisation`: coefficients with large values are normalised to `[-1, 1]` range.

### MathL2Encoder

- `test_feature_dim`: returns list of length 1, shape `(10,)`.
- `test_factored_flag`: for a product input, `is_factored_in` should be 1.0.
- `test_rational_output`: for `simplify((x²-1)/(x-1))`, `has_rational_out` should be 0.0 (result is polynomial).
- `test_epistemic_threaded`: `l1_weight` and `l1_loss` from `epistemic` arg appear correctly at indices 7 and 8.
- `test_term_ratio_clamped`: when output has many more terms, ratio is clamped to 3.0.

### MathL3Encoder

- `test_feature_dim`: returns list of length 1, shape `(12,)`.
- `test_differentiate_deg_decrease`: `diff(x³)` → `3x²`, `deg_decreased` should be 1.0.
- `test_integrate_deg_increase`: `integrate(x²)` → `x³/3`, `deg_increased` should be 1.0.
- `test_onehot_exclusive`: exactly one of `deg_increased`, `deg_decreased`, `deg_unchanged` is 1.0 for any valid pair.
- `test_factor_to_expand_flags`: `factored_to_expanded=1.0`, `expanded_to_factored=0.0` for an expand transformation.
- `test_epistemic_threaded`: `l2_weight` and `l2_loss` appear correctly at indices 10 and 11.

### Task Generation

- `test_no_degenerate_candidates`: all 5 candidates for any generated task are mutually distinct (no two SymPy-equal outputs).
- `test_correct_idx_valid`: `correct_idx` is in `range(5)` and `candidates[correct_idx]` equals `test_output`.
- `test_factor_noop_excluded`: no task has `factor(input_eq) == input_eq`.

---

## Non-Goals (this sub-project)

- No persistent or cross-task learning (per-task reset only).
- No L4 (generative output synthesis) or L5 (meta-monitoring).
- No modification of existing benchmarks or core `hpm/` modules.
- No support for multi-variable polynomials (single variable `x` only).
- No trigonometric, exponential, or logarithmic transformations (polynomial families only).
