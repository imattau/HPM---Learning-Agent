# Spec: Adaptive-K Design — CharClassAdapter + grow_latent()

**Date:** 2026-04-25  
**Status:** Proposed  
**Scope:** Two targeted additions to HPM v4 that keep latent state K small while maintaining expressivity.

---

## Background and Motivation

The three-level HMM in `HierarchicalPattern` has forward-backward cost O(T · K^6) and emission-lookup cost O(T · K^3 · obs_dim). At K=2 and obs_dim=95 (ASCII printable range), emission lookups already dominate for Wikipedia-scale sequences. Two complementary strategies address this without touching the core inference algorithm:

- **Solution 3 (this spec, Feature A):** Reduce obs_dim by preprocessing characters into coarse classes before they reach the pattern. This is a lossless compression at the input boundary.
- **Solution 4 (this spec, Feature B):** Allow K to grow from a small base only when the pattern has both structure (high compression) and unresolved loss (needs capacity). This avoids starting with a large K that wastes memory and compute on uninformative states.

Solutions 1 and 2 (replicator dynamics pruning + `compose_predictions` ensemble) are already implemented in `operators/dynamics.py` and `agents/reasoning.py` respectively and are not repeated here.

---

## Why K Must Stay Small

The existing population of `HierarchicalPattern` objects uses composition as the primary expressivity mechanism: many small-K specialists, each with a narrow obs_dim, whose predictions are blended by `compose_predictions` in the Reasoner. A single pattern with K=8 and obs_dim=95 has 8×8×3 + 8×95 = 952 free parameters; five K=2 specialists with obs_dim=5 have 5×(4×3 + 2×5) = 5×22 = 110 free parameters total — an 8.7× reduction in parameters and a much larger reduction in forward-backward cost (K^6: 64 vs 262144 per pattern per timestep).

The correct lever for Wikipedia-scale tasks is therefore obs_dim reduction (Feature A) before K growth (Feature B).

---

## Feature A: CharClassAdapter

### Location

`hpm_ai_v4/io/adapters.py` — new class added after `DiscreteInputAdapter`.

### Purpose

Maps the 95 printable ASCII character IDs (0–94, corresponding to chr(32)–chr(126)) to 5 coarse character classes, reducing obs_dim from 95 to 5. This makes B a (K × 5) matrix instead of (K × 95), reducing emission-lookup cost by 19×.

### Class Definition

```
CharClassAdapter
    Attributes:
        CLASS_NAMES: list[str] = ['letter', 'digit', 'space', 'punctuation', 'newline']
        _table: dict[int, int]  # char_id (0-94) -> class_id (0-4)

    Methods:
        encode(char_id: int) -> int
            Maps a character ID (as produced by TextAdapter.vocab, i.e. ord(ch)-32)
            to a class ID 0–4. Clamps out-of-range inputs to class 3 (punctuation).
            Class assignment:
                0 (letter):      char_id in [33,58] ∪ [65,90]  (A-Z, a-z offset by -32)
                1 (digit):       char_id in [16,25]             (0-9 offset by -32)
                2 (space):       char_id == 0                   (' ')
                3 (punctuation): all other printable ASCII
                4 (newline):     char_id == -22 (special: pass ord('\n')-32 = -22, or handle as special sentinel)

        decode_class(class_id: int) -> str
            Returns CLASS_NAMES[class_id]. Raises ValueError for out-of-range.

    Property:
        obs_dim -> int: returns 5
```

### Character Class Mapping Detail

The TextAdapter vocab maps `chr(i) -> i - 32` for i in range(32, 127), giving IDs 0–94:

| ASCII range | chars       | ID range | class |
|-------------|-------------|----------|-------|
| chr(32)     | space       | 0        | 2     |
| chr(48-57)  | 0-9         | 16-25    | 1     |
| chr(65-90)  | A-Z         | 33-58    | 0     |
| chr(97-122) | a-z         | 65-90    | 0     |
| chr(10)     | newline     | -22      | 4     |
| all others  | punctuation | various  | 3     |

Newline (ord('\n') = 10, ID = 10-32 = -22) is handled as a special case since it falls outside the printable ASCII range. Callers should pass -22 or use the sentinel value `CharClassAdapter.NEWLINE_ID = -22`.

### Composability

`CharClassAdapter` is not itself an `InputAdapter` (it does not implement `to_observations`). It is a pure encoding step composable with any upstream adapter:

```python
text_adapter = TextAdapter()
char_class = CharClassAdapter()
raw_ids = text_adapter.to_observations(text)
class_ids = [char_class.encode(c) for c in raw_ids]
# class_ids now has obs_dim=5, safe to pass to a pattern with obs_dim=5
```

This keeps the existing adapter hierarchy intact.

### Cost Reduction

- Emission matrix B: (K × 95) → (K × 5), 19× fewer entries.
- Forward-backward inner loop: `np.log(self.B[z1, obs_seq[t]])` touches a table 19× smaller, improving cache locality.
- Sufficient statistics `SS_B`: same 19× reduction.

---

## Feature B: grow_latent() on HierarchicalPattern

### Location

`hpm_ai_v4/pattern.py` — new method on `HierarchicalPattern`.

### Purpose

Allow a pattern to expand its latent dimension K → K+1 at runtime when it has demonstrated that it has learned structure (high compression) but still has high loss (needs more capacity). This implements adaptive K without requiring the caller to pre-specify a large K.

### Method Signature

```python
def grow_latent(self, noise_scale: float = 0.01) -> None:
```

### Expansion Logic

Let K = self.latent_dim. After calling grow_latent():

**A3: (K, K) → (K+1, K+1)**
- Copy existing A3 into new_A3[:K, :K]
- New row K: uniform(1/(K+1)) + noise for each of K+1 columns
- New column K for existing rows: noise only
- Re-normalise every row to sum to 1

**A32, A21: same expansion as A3**

**B: (K, obs_dim) → (K+1, obs_dim)**
- Copy existing B into new_B[:K, :]
- New row K: mean of existing rows + noise
- Re-normalise every row to sum to 1

**pi3: (K,) → (K+1,)**
- Append 1/(K+1) to existing pi3
- Re-normalise to sum to 1

**Sufficient statistics (SS_A3, SS_A32, SS_A21, SS_B):**
- Expanded to match new shapes with small prior (0.1) in new entries

**self.latent_dim:** incremented by 1

### Exact Numpy Implementation

```python
def grow_latent(self, noise_scale: float = 0.01) -> None:
    K = self.latent_dim
    K1 = K + 1
    rng = np.random.default_rng()

    def expand_transition(A):
        new_A = np.zeros((K1, K1))
        new_A[:K, :K] = A
        new_A[K, :] = 1.0 / K1 + rng.normal(0, noise_scale, K1)
        new_A[:K, K] = np.abs(rng.normal(0, noise_scale, K))
        new_A = np.abs(new_A)
        row_sums = new_A.sum(axis=1, keepdims=True)
        return new_A / (row_sums + 1e-12)

    def expand_emission(B):
        new_B = np.zeros((K1, self.obs_dim))
        new_B[:K, :] = B
        new_B[K, :] = B.mean(axis=0) + rng.normal(0, noise_scale, self.obs_dim)
        new_B = np.abs(new_B)
        row_sums = new_B.sum(axis=1, keepdims=True)
        return new_B / (row_sums + 1e-12)

    self.A3  = expand_transition(self.A3)
    self.A32 = expand_transition(self.A32)
    self.A21 = expand_transition(self.A21)
    self.B   = expand_emission(self.B)

    new_pi = np.append(self.pi3, 1.0 / K1)
    new_pi = np.abs(new_pi)
    self.pi3 = new_pi / (new_pi.sum() + 1e-12)

    # Expand sufficient statistics
    def expand_ss_transition(SS):
        new_SS = np.ones((K1, K1)) * 0.1
        new_SS[:K, :K] = SS
        return new_SS

    def expand_ss_emission(SS):
        new_SS = np.ones((K1, self.obs_dim)) * 0.1
        new_SS[:K, :] = SS
        return new_SS

    self.SS_A3  = expand_ss_transition(self.SS_A3)
    self.SS_A32 = expand_ss_transition(self.SS_A32)
    self.SS_A21 = expand_ss_transition(self.SS_A21)
    self.SS_B   = expand_ss_emission(self.SS_B)

    self.latent_dim = K1
```

### Growth Trigger in HPMAgent

A private method `_maybe_grow_patterns(max_K=8, loss_threshold=1.0)` is added to `HPMAgent` and called every 500 steps inside `perceive_and_learn`.

Trigger condition for pattern `p`:
1. `p.compression(self.obs_buffer) > 0.3` — the pattern has learned genuine hierarchical structure (MI between z1 and z2 is positive)
2. `p.running_loss > loss_threshold` — the pattern still has significant prediction error
3. `p.latent_dim < max_K` — not already at the cap

Rationale for the conjunction: compression > 0.3 without high loss means the pattern is doing well — no need to grow. High loss without compression means the pattern hasn't found structure yet — growing K without structure leads to symmetry-breaking failure, not improvement. Only when both conditions hold is capacity the bottleneck.

Max K cap at 8: beyond K=8, forward-backward cost (K^6 = 262144) per pattern per timestep becomes prohibitive for a population of patterns.

Growth is checked every 500 steps to avoid oscillation and give newly expanded patterns time to stabilise before the next growth decision.

### No New Dependencies

Both features are pure numpy. No new imports are required beyond what `pattern.py` and `adapters.py` already use.

---

## Design Constraints Summary

| Constraint | Feature A | Feature B |
|---|---|---|
| No new dependencies | Yes (pure lookup table) | Yes (pure numpy) |
| obs_dim reduction | 95 → 5 (19×) | No change |
| K kept small | Enables small K at init | Grows only under dual condition |
| Composable with existing | Yes (wraps TextAdapter output) | Yes (called from perceive_and_learn) |
| Max K enforced | N/A | Hard cap at 8 |
