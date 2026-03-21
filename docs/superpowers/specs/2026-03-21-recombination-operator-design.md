# Recombination Operator Design Spec

**Date:** 2026-03-21

## Goal

Implement the Recombination Operator (Appendix E) — a mechanism by which an HPM agent autonomously creates new candidate patterns by mating two high-level (Level 4+) patterns when structural pressure is high. Recombination is the gateway to Level 5 (Generative / Expert) pattern formation: it produces novel hypotheses that the existing selection dynamics then confirm or discard.

## Background

The current agent loop updates weights via replicator dynamics and conflict inhibition but never generates new patterns — the library can only shrink (pruning) or drift (update). When two Level 4 patterns are in sustained conflict, neither can dominate; the agent is stuck in a local structural deadlock. The Recombination Operator resolves this by synthesising a candidate `h*` from the structural "DNA" of both parents, evaluating its fit against recent experience, and admitting it to the library if it earns a positive insight score.

`GaussianPattern.recombine(other)` already exists as a 50/50 convex combination of parent parameters. This spec wires that primitive into a full operator with trigger logic, insight evaluation, and entry weight management.

## Architecture

Three components interact:

1. **`MetaPatternRule.step()` returns `StepResult(weights, total_conflict)`** — exposes the total conflict tension (`beta_c * Σ kappa_ij * w_i * w_j`) computed from the existing incompatibility matrix. No additional work; this value was already computed and discarded.

2. **`RecombinationOperator`** (`hpm/dynamics/recombination.py`) — stateless class. `attempt()` selects a pair, creates `h*`, gates on structural validity, computes the insight score, and returns a `RecombinationResult` or `None`.

3. **`Agent`** — owns the ring buffer of recent observations, the cooldown counter, and the trigger decision. Calls `operator.attempt()` when the trigger fires, saves the accepted pattern, and renormalises all weights.

## Trigger Logic

Recombination fires when **either** condition is met, subject to cooldown:

```
trigger = (t % config.T_recomb == 0) OR (total_conflict > config.conflict_threshold)
fire    = trigger AND (t - last_recomb_t >= config.recomb_cooldown)
```

- **Time trigger** (`T_recomb`): routine structural exploration — the agent periodically "ruminates" on its best patterns.
- **Conflict trigger** (`conflict_threshold`): crisis-driven — sustained incompatibility between Level 4 patterns signals that the current structural laws are irreconcilable; recombination seeks a hybrid that resolves the tension.
- **Cooldown** (`recomb_cooldown`): refractory period preventing thrashing after a trigger fires.

`total_conflict` comes from `StepResult.total_conflict`, making it a first-class signal with no additional computation.

`recombination_trigger` in the return dict is `"time"`, `"conflict"`, or `None` (not triggered).

## Recombination Operator

### `RecombinationResult`

```python
from dataclasses import dataclass

@dataclass
class RecombinationResult:
    pattern: GaussianPattern
    insight_score: float
    parent_a_id: str
    parent_b_id: str
    trigger: str   # "time" | "conflict"
```

### `RecombinationOperator.attempt()`

```python
def attempt(
    self,
    patterns: list,
    weights: np.ndarray,
    obs_buffer: list,       # list of np.ndarray, recent observations
    config: AgentConfig,
    trigger: str,
) -> RecombinationResult | None:
```

**Internal sequence (per attempt, up to `N_recomb` pair draws):**

1. **Level gate:** `candidates = [p for p in patterns if p.level >= 4]`. Return `None` if `len(candidates) < 2`.

2. **Pair sampling:** Build joint weights `score_ab = w_a * w_b` for all Level 4+ pairs. Apply softmax with temperature `recomb_temp`: `p_ab = softmax(score_ab / recomb_temp)`. Draw a pair. Reject if `sym_kl_normalised(parent_a, parent_b) >= config.kappa_max` — incompatible parents produce incoherent children. Up to `N_recomb` draws; return `None` if all rejected.

3. **Crossover:** `h* = parent_a.recombine(parent_b)` (existing convex combination).

4. **Feasibility gate:** `if not h*.is_structurally_valid(): try next draw`. This check runs **before** the insight computation to avoid scoring incoherent candidates.

5. **Insight score:**
   - `Nov(h*) = max(sym_kl_normalised(h*, parent_a), sym_kl_normalised(h*, parent_b))` — in [0, 1]; high novelty means h* is maximally distant from at least one parent. (Using `max` rather than `1 - max`: `sym_kl_normalised` returns values near 1 for maximally divergent distributions, so no inversion is needed.)
   - `Eff(h*) = mean([-h*.log_prob(x) for x in obs_buffer])` — `log_prob(x)` returns the NLL (positive loss); negating gives the mean log-likelihood (a negative number, higher = better fit). Empty buffer returns `0.0`. Eff is typically negative; `I(h*) > 0` requires that novelty is large enough to offset negative efficacy, so only children that are both novel and well-fitting are admitted. **Empty-buffer edge case:** when `obs_buffer` is empty, `Eff = 0.0` and `I(h*) = beta_orig * alpha_nov * Nov >= 0`, meaning any novel, structurally valid child is accepted unconditionally. This is intended cold-start behaviour — the agent has no observations yet to use for efficacy gating, so novelty alone is sufficient.
   - `I(h*) = config.beta_orig * (config.alpha_nov * Nov + config.alpha_eff * Eff)`

6. **Accept/discard:** Return `None` if `I(h*) <= 0`. Otherwise return `RecombinationResult`.

### Pair sampling detail

`sym_kl_normalised` (already in `meta_pattern_rule.py`) is reused for both the pair rejection check (`kappa_ab >= kappa_max`) and the novelty computation (`kappa_{h*,parent}`). The operator imports it from the same module.

`recomb_temp = 1.0` (default) reduces to proportional sampling `w_a * w_b`. Lower values concentrate draws on the highest-weight pair; higher values approach uniform sampling over Level 4+ candidates.

## Agent Changes

### `__init__`

```python
from collections import deque
from ..dynamics.recombination import RecombinationOperator

self._obs_buffer: deque = deque(maxlen=config.obs_buffer_size)
self._last_recomb_t: int = -config.recomb_cooldown   # allow trigger on step 1 if desired
self._recomb_op = RecombinationOperator()
```

### `step(x)`

**At the very start of step (before evaluators):**
```python
self._obs_buffer.append(x)
```

**After dynamics, before prune loop:**
```python
step_result = self.dynamics.step(
    patterns, weights, totals,
    densities=densities,
    kappa_d_per_pattern=kappa_d_per_pattern,
)
new_weights = step_result.weights
total_conflict = step_result.total_conflict
```

**After prune loop and field registration:**
```python
recomb_result = None
recomb_attempted = False
recomb_trigger = None

time_trigger = (self._t % self.config.T_recomb == 0)
conflict_trigger = (total_conflict > self.config.conflict_threshold)
cooldown_ok = (self._t - self._last_recomb_t >= self.config.recomb_cooldown)

if (time_trigger or conflict_trigger) and cooldown_ok:
    recomb_trigger = "conflict" if conflict_trigger else "time"
    recomb_attempted = True
    # reload patterns post-prune for recombination
    post_prune_records = self.store.query(self.agent_id)
    post_prune_patterns = [p for p, _ in post_prune_records]
    post_prune_weights = np.array([w for _, w in post_prune_records])
    recomb_result = self._recomb_op.attempt(
        post_prune_patterns, post_prune_weights,
        list(self._obs_buffer), self.config, recomb_trigger,
    )
    if recomb_result is not None:
        entry_weight = self.config.kappa_0 * recomb_result.insight_score
        self.store.save(recomb_result.pattern, entry_weight, self.agent_id)
        # Renormalise all weights so they sum to 1.0
        all_records = self.store.query(self.agent_id)
        total_w = sum(w for _, w in all_records)
        if total_w > 0:
            for p, w in all_records:
                self.store.update_weight(p.id, w / total_w)
    self._last_recomb_t = self._t
```

**Note:** Recombination operates on the **post-prune** pattern population (freshly queried from the store) so that pruned patterns cannot be selected as parents.

**Step ordering within `Agent.step()`** (relevant to trigger arithmetic):
1. `self._obs_buffer.append(x)` — top of step
2. Epistemic / affective / social evaluation
3. `step_result = self.dynamics.step(...)` → unpack `new_weights`, `total_conflict`
4. Prune loop + field registration
5. `self._t += 1`  ← existing increment location
6. Recombination block (uses the incremented `self._t`)

So `self._t` at recombination time equals the number of completed steps. The time trigger `self._t % T_recomb == 0` fires first at step `T_recomb` (after the first `T_recomb` observations have been processed). With `_last_recomb_t = -recomb_cooldown`, cooldown is satisfied from step 1 onward.

**Cooldown resets on failed attempts** — `self._last_recomb_t = self._t` is assigned unconditionally when the trigger fires, regardless of whether `attempt()` returns a result. This is intentional: repeated conflict-triggered attempts on an incompatible pattern population would thrash; the cooldown enforces a refractory period even after unsuccessful attempts.

**Note:** `StepResult` is used for the return value of `MetaPatternRule.step()`. All existing callers of `step()` in `agent.py` must unpack `.weights` from the result.

### Return dict additions

```python
'total_conflict': float(total_conflict),
'recombination_attempted': recomb_attempted,
'recombination_accepted': recomb_result is not None,
'recombination_trigger': recomb_trigger,
'insight_score': recomb_result.insight_score if recomb_result else None,
'recomb_parent_ids': (recomb_result.parent_a_id, recomb_result.parent_b_id) if recomb_result else None,
```

## Components

| File | Change |
|---|---|
| `hpm/dynamics/recombination.py` | CREATE — `RecombinationResult`, `RecombinationOperator` |
| `hpm/dynamics/meta_pattern_rule.py` | MODIFY — `StepResult` NamedTuple, `step()` returns it |
| `hpm/dynamics/__init__.py` | MODIFY — re-export `RecombinationOperator` if exists |
| `hpm/config.py` | MODIFY — 11 new fields |
| `hpm/agents/agent.py` | MODIFY — buffer, trigger, operator call, renormalise, return dict |

`hpm/store/sqlite.py` and `hpm/store/memory.py` — no changes. `update_weight()` already exists on both stores.

## New Config Parameters

| Field | Default | Meaning |
|---|---|---|
| `T_recomb` | 100 | Steps between time-triggered recombinations |
| `N_recomb` | 3 | Max pair draw attempts per trigger |
| `kappa_max` | 0.5 | Max incompatibility for eligible pair |
| `conflict_threshold` | 0.1 | `total_conflict` level that fires secondary trigger |
| `recomb_cooldown` | 10 | Min steps between any two recombinations |
| `obs_buffer_size` | 50 | Ring buffer capacity (recent observations) |
| `beta_orig` | 1.0 | Insight score scale |
| `alpha_nov` | 0.5 | Novelty weight in I(h*) |
| `alpha_eff` | 0.5 | Efficacy weight in I(h*) |
| `kappa_0` | 0.1 | Entry weight scale for accepted h* |
| `recomb_temp` | 1.0 | Softmax temperature for pair sampling |

All defaults are backward-compatible: `T_recomb=100` means recombination never fires in short test runs (< 100 steps); `conflict_threshold=0.1` is above typical single-pattern conflict (zero cross-terms); `kappa_0=0.1` is small enough not to distort existing weights significantly.

## Backward Compatibility

- `MetaPatternRule.step()` return type changes from `np.ndarray` to `StepResult`. Call sites requiring `.weights` unpacking: `agent.py` (one call) and all direct calls in `tests/dynamics/test_meta_pattern_rule.py` and `tests/dynamics/test_meta_pattern_rule_density.py`. Those test files must be updated to unpack `.weights` from the result.
- All new config fields have defaults — existing `AgentConfig(agent_id=..., feature_dim=...)` instantiations are unchanged.
- Recombination does not fire in existing tests (fewer than 100 steps, no conflict above threshold) — all 173 existing tests pass unchanged.

## Testing

### Unit: `tests/dynamics/test_recombination.py`

- `test_returns_none_when_fewer_than_two_level4_patterns` — only Level 1–3 patterns → None
- `test_returns_none_when_all_pairs_exceed_kappa_max` — two Level 4+ patterns with high KL → None after N_recomb attempts
- `test_feasibility_gate_rejects_invalid_sigma` — patch `recombine()` to return pattern with `is_structurally_valid() == False` → None
- `test_insight_score_positive_for_well_fitting_child` — parents with low log-prob on buffer → I(h*) > 0 → result returned
- `test_insight_score_zero_or_negative_discards` — buffer observations far from child → None
- `test_novelty_one_when_child_maximally_distant` — child in orthogonal subspace → Nov ≈ 1
- `test_softmax_temperature_low_concentrates_on_highest_weight_pair` — temp → 0 → highest-weight pair always drawn
- `test_empty_buffer_eff_is_zero` — obs_buffer=[] → Eff = 0.0, only novelty contributes

### Unit: `tests/dynamics/test_step_result.py` (or add to existing density test file)

- `test_step_returns_step_result` — result has `.weights` and `.total_conflict` attributes
- `test_total_conflict_zero_for_single_pattern` — single pattern → no cross terms → 0.0
- `test_total_conflict_positive_for_incompatible_patterns` — two distant patterns → total_conflict > 0

### Integration: `tests/agents/test_agent_recombination.py`

- `test_return_dict_has_recombination_keys_every_step` — all six new keys present even when not triggered
- `test_recombination_not_attempted_before_T_recomb` — configure `conflict_threshold=float('inf')` to suppress the conflict trigger; step `T_recomb-1` times → `recombination_attempted=False` on every step
- `test_time_trigger_fires_at_T_recomb` — agent with T_recomb=5, Level 4 patterns forced → attempted at step 5
- `test_conflict_trigger_fires_on_high_tension` — patch total_conflict to exceed threshold → attempted before time limit
- `test_cooldown_blocks_double_trigger` — conflict fires at step 1, conflict again at step 2 → blocked by cooldown
- `test_accepted_pattern_added_to_store` — after acceptance, store has one more pattern
- `test_weights_sum_to_one_after_acceptance` — sum of all weights == 1.0 after h* is saved
- `test_recomb_parent_ids_none_when_not_attempted` — no trigger → `recomb_parent_ids` is None
- `test_total_conflict_in_return_dict` — `total_conflict` key present and non-negative

## What This Is Not

- Not a hierarchical parent-child graph — recombined patterns are independent library entries
- Not a mutation operator — `recombine()` is a convex combination of two parents, not random perturbation
- Not a permanent Level 5 upgrade — h* starts at Level 1 and earns level through the existing classification pipeline over subsequent steps
