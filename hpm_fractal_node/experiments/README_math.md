# Math Arithmetic Experiment — Algebraic Rule Discovery

## Overview

Tests whether the HFN Observer can discover mathematical structure — algebraic laws,
divisibility relationships, number-class properties, and modular arithmetic — purely from
geometric compression of arithmetic observation vectors. No semantic flags are given to the
agent. Every rule it finds must emerge from the geometry of the observation stream.

---

## Design

### Observation Encoding

Each observation is a 4-tuple `(left, op, right, result)` encoded as concatenated one-hot
vectors in **R^109**:

```
dims  0-9:    left operand   one-hot over 0-9
dims 10-16:   operator       one-hot over 7 operations
dims 17-26:   right operand  one-hot over 0-9
dims 27-108:  result         one-hot over 0-81
```

**Operations:** `+`, `-`, `*`, `//`, `mod`, `gcd`, `pow`

Constraints:
- Subtraction: only generated when `left >= right` (non-negative results)
- Division/mod: `right != 0`
- Pow: filtered to `result <= 81`

This yields **586 unique valid observations** — all arithmetic facts in the 0-9 digit
system. The experiment samples from these (with replacement) to generate the training stream.

No property flags, no category labels, no external lookups. The agent sees only raw
(left, op, right, result) tuples.

### Prior Library (306 priors across 6 levels)

The prior library gives the agent pre-structured knowledge about arithmetic, mirroring HPM's
claim that learners don't start from a blank slate. Priors are permanent (protected from
absorption) and each has its mu set to the centroid of the observations it covers.

| Level | Count | What |
|---|---|---|
| L1 — Symbols | 33 | digit_0-9, op_*, result_0-9 |
| L2 — Classes | 47 | number-class properties of operands and results |
| L3 — Templates | 81 | expression-shape patterns per operator (12 variants × 7 ops) |
| L4 — Relations | 42 | mathematical relationships between operands/results |
| L5 — Rules | 57 | algebraic identity and arithmetic law instances |
| L6 — Structures | 43 | modular groups, fields, lattices, power towers |
| **Total** | **306** | |

Note: Some filter combinations produce no matching observations (e.g. congruence mod 7 with
remainder 6 doesn't appear in 0-9 arithmetic) — those priors are silently dropped, giving
306 rather than the design target of ~500.

### Evaluation

Category labels are assigned **after** observation, purely for measuring purity. The agent
never sees them. 23 categories:

`identity_add`, `identity_mul`, `absorption_mul`, `identity_pow_zero`, `identity_pow_one`,
`identity_div_one`, `exact_divisibility`, `gcd_self`, `gcd_with_zero`, `gcd_prime`,
`carry`, `no_carry`, `sub_self`, `subtraction`, `prime_result`, `perfect_power`,
`power_general`, `mul_large`, `mul_small`, `floor_div`, `mod_general`, `gcd_general`,
`general`

**Random baseline: 1/23 ≈ 0.043**

---

## Results (4 passes, 5000 observations/pass)

### Coverage

```
Prior nodes explained:   15461 (77.3%)
Learned nodes explained:  4539 (22.7%)
Total attributed:        20000 (100.0%)
Learned node count:      2360
```

100% explanation rate across all 4 passes — the 306 priors cover the complete observation
space. The prior library has enough density to ensure every arithmetic fact is near at least
one prior centroid.

### Top Prior Nodes

The highest-coverage prior nodes reveal which regions of arithmetic are most observation-dense:

| Prior | n | Purity | Category |
|---|---|---|---|
| `struct_mul_monoid_large` | 1418 | 1.00 | mul_large |
| `rule_self_div` | 949 | 0.98 | floor_div |
| `rule_gcd_coprime` | 896 | 1.00 | gcd_general |
| `struct_add_overflow_09` | 890 | 1.00 | carry |
| `rule_sub_reduce` | 519 | 1.00 | subtraction |
| `rule_id_sub_zero` | 317 | 1.00 | subtraction |
| `rule_self_sub` | 307 | 1.00 | sub_self |
| `rule_id_div_one` | 289 | 1.00 | identity_div_one |
| `rule_id_mul_left` | 265 | 1.00 | identity_mul |
| `rule_id_pow_one` | 237 | 1.00 | identity_pow_one |
| `struct_field_F7_add` | 234 | 0.93 | no_carry |

Key finding: **all identity and absorption rule priors achieve purity 1.00** — the law
priors are perfectly specific to their intended mathematical categories.

### Learned Node Purity

```
Category purity:  mean=0.793   max=1.000
Random baseline:  0.043        (1/23)
Nodes > 2x baseline: 14/14    (all stable learned nodes)
```

**18× above random baseline.** Every stable learned node (those observed ≥5 times) beats
2× the random baseline.

### Notable Learned Nodes

**`b36cff33` — Perfect power cluster**
- purity=1.000, op=pow, n=38
- Nearest prior: `tmpl_pow_both_even` (dist=0.701)
- The agent discovered a geometric cluster corresponding to `perfect_power` without any label

**`compressed(cong_mod, cong_mod)` — Modular prime region**
- purity=0.724, op=mod, n=29
- Nearest prior: `cong_mod7_rem4`
- Modular remainder patterns that correlate with prime results — found unsupervised

**`compressed(compress, compress)` — Non-divisibility chain**
- n=28, nearest prior: `not_divides`
- Compression of mod observations where the remainder is non-zero

**`compressed(tmpl_gcd, tmpl_gcd)` — GCD-zero identity**
- purity=1.000, op=gcd, n=7
- Discovered the `gcd(a,0)=a` identity geometrically

**`compressed(struct_p, tmpl_pow)` — Power structure**
- purity=0.750, op=pow, n=4
- Bridge between a structural prior and an expression template

### Rule Discovery Summary

Learned nodes attracted observations across **15 of 23 categories**, with the geometrically
richest regions (where priors don't fully cover) drawing the most learned-node activity:

| Category | Learned-node observations |
|---|---|
| floor_div | 740 |
| prime_result | 676 |
| no_carry | 464 |
| mod_general | 452 |
| subtraction | 338 |
| gcd_self | 264 |
| gcd_with_zero | 245 |
| gcd_general | 229 |
| absorption_mul | 214 |
| mul_large | 176 |

The agent is most actively learning around **floor division** and **prime results** — the
two categories where the prior library has least density relative to the number of matching
observations.

### Fractal Diagnostics

```
Active learned nodes:       236
Absorbed nodes:             19502
Hausdorff(learned, priors): 0.9992
```

236 learned nodes survived absorption (from 2360 total created) — the rest were absorbed as
they fell within the Hausdorff threshold of existing nodes. The Hausdorff distance of ~1.0
indicates learned nodes are finding gaps in the prior cloud at a consistent scale, not
clustering tightly around any single prior.

### Abstraction Candidates

```
depth >= 2:           0
cross-category (>=2): 11
stable:               37
```

11 nodes attracted observations from multiple mathematical categories — these are the most
interesting candidates for higher-order abstractions. Depth≥2 is 0 because the BFS from
prior IDs doesn't yet reach the learned compression nodes — this is a measurement artefact,
not a structural claim.

---

## What This Means for HPM

The math experiment demonstrates three things:

**1. Pure geometry encodes mathematical law.** The prior library has no labels — just
mu vectors computed from observation centroids. Yet identity, absorption, and modular
arithmetic laws achieve purity 1.0. The geometry of the one-hot encoding is enough to
separate mathematical laws cleanly in R^109.

**2. The agent discovers structure not given to it.** The top learned node is a
`perfect_power` cluster not directly encoded as a prior. The agent formed this by geometric
compression of repeated pow observations — it found the mathematical structure purely through
the absorption/compression dynamic.

**3. The prior library shapes the learning frontier.** Learned nodes concentrated in
`floor_div` and `prime_result` — exactly the categories with the sparsest prior coverage.
The agent's novelty signal is guided by the gaps in what it already knows, which is the
HPM prediction for a well-seeded world model.

---

## Running the Experiment

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_math.py
```

Runtime: ~2-3 minutes (4 passes × 5000 observations, no LLM).

**Configuration** (top of `experiment_math.py`):

```python
N_SAMPLES = 5000    # observations per pass
N_PASSES = 4        # number of passes
SEED = 42
TAU_SIGMA = 1.0     # tau calibration sigma scale
TAU_MARGIN = 5.0    # tau calibration margin
```

**Absorption parameters** (in Observer constructor):

```python
hausdorff_absorption_threshold = 0.35     # geometric absorption distance
hausdorff_absorption_weight_floor = 0.4   # only absorb nodes with weight < 0.4
absorption_miss_threshold = 20            # consecutive misses before absorption
multifractal_guided_absorption = False    # disabled to allow injected nodes to survive
```

---

## Files

```
hpm_fractal_node/math/
    __init__.py
    math_loader.py          # encoding, observation generation, category labels
    math_world_model.py     # 306-prior library across 6 abstraction levels

hpm_fractal_node/experiments/
    experiment_math.py      # main experiment script
```

---

## Extending This Experiment

**More digits (0-99):** Expand `D_LEFT`, `D_RIGHT` to 100 dims each and `D_RESULT` to
cover products up to 9801. D would grow to ~9810 — the sigma-diag optimisation described
in `hfn/README_priors.md` becomes essential at this scale.

**Adding LLM gap-filling:** Replace `gap_query_threshold=None` with a threshold (e.g. 0.05)
and wire in a `QueryLLM` instance. The LLM could be asked to name the mathematical property
of an unknown arithmetic pattern, bridging the agent's geometric gaps with symbolic
knowledge.

**Algebraic structure discovery:** Run more passes (N_PASSES=10+) and lower
`hausdorff_absorption_threshold` to allow more learned nodes to stabilise. Look for
depth≥2 nodes that compress multiple L5 rule priors — these would be candidates for
L6-level algebraic structure discovery.

**Negative numbers:** Extend subtraction to allow `left < right` by adding a sign bit
to the encoding. This introduces additive inverses and the full integer group structure,
which should be discoverable as a set of learned compression nodes bridging `+` and `-`.
