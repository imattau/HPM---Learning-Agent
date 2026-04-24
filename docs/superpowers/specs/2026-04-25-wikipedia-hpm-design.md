# Wikipedia Character-Stream Simulation — Design Spec (v2)

**Date**: 2026-04-25
**Branch**: hpm-ai-v4-dev
**Status**: Approved for implementation
**Supersedes**: `docs/superpowers/specs/2026-04-24-wikipedia-simulation-design.md`

---

## 0. Architecture Principle: Small-K, Deep Hierarchy

HPM uses **small K (max 4)** per pattern, with **depth** (multiple levels) providing complexity.
A population of 10–15 specialised K=2 patterns beats one K=16 monolith.

This is the fundamental design constraint for all patterns in this simulation.
**Hard limit: K=2 for all patterns in this simulation. grow_latent() cap: K=4.**

The previous spec (2026-04-24-wikipedia-simulation-design.md) used `latent_dim=16` and `obs_dim=96` on a single `HPMAgent`. This is **architecturally incorrect** under HPM and is superseded entirely.

---

## 1. Goal

Validate that a 3-level HPM population with small-K patterns can learn hierarchical linguistic
structure from a raw character stream. The simulation must expose that structure through
programmatic reasoning queries (char-class IDs in, class probabilities out).

Specifically:
- Level-1 patterns achieve next char-class prediction accuracy > 60% (baseline: 1/5 = 20%).
- Level-2 compression (MI between L1 and L2 latent states) rises from ~0 to > 0.2 by step 50,000.
- Level-3 compression rises from ~0 to > 0.1 by step 100,000.
- Planning to a space-class within a short horizon succeeds > 60% of the time.

---

## 2. CharClassAdapter

**File**: `hpm_ai_v4/io/adapters.py` (ADD to existing file)

Note: a `CharClassAdapter` already exists in `adapters.py`. The spec describes its intended
behaviour and any additions needed. The existing implementation maps `ord(ch) - 32` IDs to
5 classes; the simulation uses raw characters as input, so `encode(ch: str) -> int` is the
primary interface for the stream.

```
Classes:
    0 = letter      (a-z, A-Z)
    1 = digit       (0-9)
    2 = space       (' ')
    3 = punctuation (all other printable ASCII 32-126)
    4 = newline     ('\n')
```

The existing `CharClassAdapter.encode(char_id: int)` takes an already-shifted ID. The simulation
will call it with `ord(ch) - 32` for printable ASCII, or the sentinel `-22` for newlines. No new
class is needed — the existing implementation is sufficient.

---

## 3. WikipediaStream

**File**: `hpm_ai_v4/simulations/wikipedia_sim.py`

```python
class WikipediaStream:
    def __init__(self, filepath: str, adapter: CharClassAdapter):
        ...
    def __iter__(self) -> Iterator[int]:
        # Reads file char by char; converts via adapter; loops on exhaustion.
        # Yields class IDs in [0, 4].
        ...
```

Behaviour:
- Reads UTF-8 file character by character.
- For each character, computes `char_id = ord(ch) - 32` for printable ASCII 32–126,
  `char_id = -22` (sentinel) for `'\n'`, and skips all other characters.
- Passes `char_id` to `adapter.encode(char_id)` to get class ID 0–4.
- Loops back to start when file is exhausted.

---

## 4. 3-Level Population Architecture

### Level 1 — Feature Detectors (10 patterns, K=2, obs_dim=5)

**Input**: CharClassAdapter output — class ID in {0, 1, 2, 3, 4}.

Each pattern is a `HierarchicalPattern(latent_dim=2, obs_dim=5)`. Ten patterns initialised with
different random seeds specialise via replicator dynamics into distinct feature detectors:

| Pattern | Expected specialisation |
|---|---|
| P1 | letter vs non-letter |
| P2 | space vs non-space |
| P3 | punctuation vs non-punctuation |
| P4 | vowel-class vs consonant-class (within letter class) |
| P5–P10 | various transition biases (repeated-class runs, alternation, etc.) |

B matrix shape: (2×5) — trivially fast EM. A3, A32, A21 all (2×2).

A separate **PatternField** tracks population-level frequencies for Level-1 replicator dynamics.

### Level 2 — Chunk Detectors (5 patterns, K=2, obs_dim=2)

**Input**: SEQUENCE of Level-1 latent state (argmax of `alpha[-1]` from the best Level-1 pattern,
yielding a binary symbol 0 or 1).

Each pattern is a `HierarchicalPattern(latent_dim=2, obs_dim=2)`. Five patterns specialise into:

| Pattern | Expected specialisation |
|---|---|
| P1 | word-start detector (space→letter transition) |
| P2 | word-end detector (letter→space transition) |
| P3 | space-run detector (space→space) |
| P4–P5 | other short-sequence regularities |

B matrix shape: (2×2). All transition matrices (2×2).

### Level 3 — Grammar Patterns (3 patterns, K=2, obs_dim=2)

**Input**: SEQUENCE of Level-2 latent state (argmax of `alpha[-1]` from the best Level-2 pattern).

Three patterns learn transitions between word-level units (e.g. word-boundary sequences,
punctuation clusters, paragraph structure).

B matrix shape: (2×2).

### Inter-level interface: get_top_state()

Each level feeds its most-probable latent state to the next level. This is a causal chain,
not a joint model:

```python
def get_top_state(self, obs_seq: List[int]) -> int:
    """Run forward algorithm on obs_seq; return argmax of alpha[-1] marginalised over z3,z2."""
    ...
```

Returns an int in `{0, ..., K-1}`. This is the "upward pass" — lower-level output becomes
upper-level input.

**Implementation note**: `get_top_state` runs the forward-only pass (no backward), marginalises
`alpha[-1]` over all but the outermost latent dimension (z3), and returns `argmax`. This is
`O(K^3 * T)` — negligible for K=2, T=20.

---

## 5. Training Loop (per character step)

```
1. Env emits char → CharClassAdapter → class_id in {0,1,2,3,4}
2. All Level-1 patterns: observe(class_id) + adapt(L1_buffer[-20:])
3. L1 field_freq update → replicator dynamics on L1 population
4. Extract L1 latent: best_L1.get_top_state(L1_buffer[-20:]) → state_id in {0,1}
5. Append state_id to L2_buffer
6. All Level-2 patterns: observe(state_id) + adapt(L2_buffer[-20:])
7. L2 field_freq update → replicator dynamics on L2 population
8. Extract L2 latent: best_L2.get_top_state(L2_buffer[-20:]) → state_id in {0,1}
9. Append state_id to L3_buffer
10. All Level-3 patterns: observe(state_id) + adapt(L3_buffer[-20:])
11. L3 field_freq update → replicator dynamics on L3 population
```

Replicator dynamics runs **separately per level** (three independent populations). Recombination
is **within-level only** — K stays the same after recombine.

**best_L1** = pattern with highest replicator weight in L1 population.

### grow_latent() cap

`grow_latent()` is called at most once per 500 steps, and only if `latent_dim < 4`.
`HierarchicalPattern` constructor logs a warning if `latent_dim > 4`.

---

## 6. Programmatic Reasoning Interface (TextReasoningInterface)

**File**: `hpm_ai_v4/simulations/text_reasoning.py`

All queries are char-class IDs in/out — NO natural language generation.

```python
class TextReasoningInterface:
    def __init__(self, L1_patterns, L2_patterns, L3_patterns,
                 L1_reasoner: Reasoner, L2_reasoner: Reasoner, L3_reasoner: Reasoner):
        ...

    def next_char_predict(self, prefix_str: str) -> List[Tuple[str, float]]:
        """
        Encode prefix via CharClassAdapter → class IDs.
        Run compose_predictions on Level-1 population.
        Return top-5 CLASS predictions with probs.
        Example: [("letter", 0.7), ("space", 0.2), ("digit", 0.05), ...]
        Cannot predict specific characters — only character classes.
        """

    def word_boundary_predict(self, prefix_str: str) -> float:
        """
        Encode prefix → L1 latent sequence → run compose_predictions on L2 population.
        Return P(next_L2_state = word_boundary_state).
        """

    def plan_to_space(self, horizon: int, num_rollouts: int) -> List[str]:
        """
        Level-3 Reasoner.plan(goal_state=space_class_id, horizon=horizon).
        Returns sequence of class IDs decoded to class names.
        e.g. ["letter", "letter", "space"]
        """

    def counterfactual_shift(self, context: str, forced_class: int) -> List[Tuple[str, float]]:
        """
        Level-1 Reasoner.counterfactual(encoded_context, forced_class).
        Returns top-5 (class_name, prob) after forcing the specified class.
        """
```

### Type summary

| Method | Input | Return |
|---|---|---|
| `next_char_predict` | `str` | `List[Tuple[str, float]]` (len=5) |
| `word_boundary_predict` | `str` | `float` |
| `plan_to_space` | `int, int` | `List[str]` |
| `counterfactual_shift` | `str, int` | `List[Tuple[str, float]]` (len=5) |

---

## 7. K Constraints (HARD LIMITS)

| Constraint | Value |
|---|---|
| Max K per pattern in this simulation | 2 (initial) |
| grow_latent() cap | 4 |
| obs_dim Level 1 | 5 |
| obs_dim Level 2 | 2 |
| obs_dim Level 3 | 2 |

`HierarchicalPattern.__init__` must emit `warnings.warn` if `latent_dim > 4`.

---

## 8. Metrics and Success Criteria

All metrics measured on a held-out 1,000-step window not seen during training.

| Metric | Definition | Threshold |
|---|---|---|
| L1 prediction accuracy | `argmax(next_char_predict)` matches ground-truth class | > 60% (baseline 20%) |
| L2 compression (MI) | `best_L2.compression(L2_buffer)` at step 50k | > 0.2 nats |
| L3 compression (MI) | `best_L3.compression(L3_buffer)` at step 100k | > 0.1 nats |
| Planning success | `plan_to_space` reaches space-class within horizon | > 60% |

---

## 9. Files

| File | Action | Purpose |
|---|---|---|
| `hpm_ai_v4/io/adapters.py` | MODIFY | Add `encode(char: str) -> int` convenience method to `CharClassAdapter` |
| `hpm_ai_v4/pattern.py` | MODIFY | Add `get_top_state(obs_seq)` method; add K>4 constructor warning |
| `hpm_ai_v4/simulations/wikipedia_sim.py` | CREATE | `WikipediaStream` + `run_simulation()` |
| `hpm_ai_v4/simulations/text_reasoning.py` | CREATE | `TextReasoningInterface` |
| `hpm_ai_v4/simulations/data/get_corpus.py` | CREATE | Download Wikipedia sample to `data/wiki_sample.txt` |
| `hpm_ai_v4/tests/test_wikipedia_sim.py` | CREATE | Unit tests for all above |

No new pattern classes are introduced. The 3-level architecture is implemented as three
independent populations of `HierarchicalPattern` instances connected by `get_top_state()`.

---

## 10. Out of Scope

- Natural language query interface
- GPU acceleration (numpy-only)
- Multi-agent social evaluators (single 3-level agent for this validation)
- `latent_dim > 2` at initialisation
- Joint 3-level HMM (causal chain is sufficient)
- Any K > 4 after adaptive growth
