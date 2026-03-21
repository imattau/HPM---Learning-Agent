# Negative Patterns / Inhibitory Tier — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Touches:** `hpm/store/tiered_store.py`, `hpm/field/field.py`, `hpm/agents/agent.py`, `hpm/config.py`, `benchmarks/multi_agent_arc.py`

---

## 1. Motivation

### 1.1 The Greedy Deletion Problem in ARC Persistent Mode

In the current `run_persistent` implementation (`benchmarks/multi_agent_arc.py`), `end_context(correct=False)` simply discards all Tier 1 patterns for that task. This is a missed opportunity: patterns that were *active during a failed task* carry information. Specifically, if a failed task's Tier 1 patterns are highly similar to established Tier 2 positive patterns, that conflict is evidence that the Tier 2 pattern is being misapplied — or that certain structural regions of feature space are anti-predictive.

Currently, nothing learns from failure. Pattern histories from incorrect tasks evaporate.

### 1.2 What Inhibitory Patterns Represent (HPM framing)

Within the HPM framework, patterns are not merely statistical summaries of observations — they are **evaluated structural regularities** that are selected or suppressed by evaluators. An inhibitory (negative) pattern is a pattern that has been reinforced by *failure signal*: the system has learned that a particular region of feature space, when matched, predicts *wrong* outcomes.

In HPM terms:
- **Positive Tier 2 patterns** are meta-patterns that have survived similarity-merge on correct tasks. They represent structural regularities that generalise across successes.
- **Negative Tier 2 patterns** are inhibitory meta-patterns promoted when a failed task's Tier 1 patterns strongly conflict with existing Tier 2 positives. They represent structural regions that are misleadingly similar to success patterns but in fact anti-predictive.

### 1.3 Design Goal

Add an inhibitory tier to the store and scoring system so that:
1. Failed tasks contribute taboo knowledge (what *not* to predict).
2. The ensemble scoring function penalises candidates that resemble negative patterns.
3. The PatternField propagates inhibitory signals across agents (social taboo propagation).
4. StructuralLawMonitor gains visibility into the negative pattern population for diagnostic purposes.

---

## 2. Architecture Overview

Four coordinated changes are required:

| Section | Component | Change |
|---------|-----------|--------|
| §3 | `TieredStore` | Add `_tier2_negative` partition + `query_negative` + `negative_merge` |
| §4 | `PatternField` | Add inhibitory broadcast/pull channel |
| §5 | `ensemble_score` (ARC) | Subtract negative NLL from total score |
| §6 | Lifecycle & monitoring | `end_context` routing, `StructuralLawMonitor` metrics, Fear Reset |

`GaussianPattern` is **not modified**. Polarity is determined entirely by which store partition a pattern lives in, not by any field on the pattern object itself.

---

## 3. Section 1: Core Data Model — TieredStore Changes

### 3.1 New Partition

`TieredStore.__init__` gains one new attribute:

```python
self._tier2_negative: InMemoryStore = InMemoryStore()
```

This is a peer of `self._tier2` (the existing positive Tier 2 store). It holds patterns that have been promoted from failed-task Tier 1 contexts because they conflict with established positive Tier 2 patterns.

The existing `_tier1`, `_tier2`, and `_current_context` attributes are unchanged.

### 3.2 New Method: `query_negative`

```python
def query_negative(self, agent_id: str) -> list[tuple[GaussianPattern, float]]:
    """Return all negative Tier 2 patterns for agent_id."""
    return self._tier2_negative.query(agent_id)
```

Returns `list[(pattern, weight)]` in the same format as `query()`. Called by `ensemble_score` and by `PatternField` inhibitory broadcast.

### 3.3 New Method: `negative_merge`

```python
def negative_merge(
    self,
    context_id: str,
    conflict_threshold: float = 0.7,
    max_tier2_negative: int = 100,
) -> None:
```

**Algorithm:**

1. Retrieve all Tier 1 records for `context_id` from `self._tier1[context_id]`.
2. Retrieve all positive Tier 2 records from `self._tier2`.
3. For each Tier 1 pattern `p1` (with mean vector `mu1`):
   a. Compute cosine similarity between `mu1` and each positive Tier 2 pattern `mu2`.
   b. Find `best_sim = max(cosine_similarity(mu1, mu2_j) for all j)`.
   c. If `best_sim >= conflict_threshold` AND `len(_tier2_negative) < max_tier2_negative`:
      - Promote `p1` to `_tier2_negative` with weight `w1 * 0.5` (halved entry weight, matching `similarity_merge` convention).
   d. Else: discard (genuine noise, no conflict detected).
4. Zero-norm vectors (norm < 1e-8) are skipped (same guard as `similarity_merge`).

**Conflict rationale:** A Tier 1 pattern from a *failed* task that is highly similar (cosine ≥ 0.7) to an established *positive* Tier 2 pattern represents a structural confusion — the agent applied a known-good pattern to the wrong context. Promoting it to `_tier2_negative` encodes the taboo: "this pattern shape, in this context, was wrong."

Patterns below `conflict_threshold` are discarded because they do not represent a specific conflict with known good patterns — they are just noise from a task that didn't work.

### 3.4 Updated `end_context`

```python
def end_context(self, context_id: str, correct: bool) -> None:
    if correct and context_id in self._tier1:
        self.similarity_merge(context_id)
    elif not correct and context_id in self._tier1:
        self.negative_merge(context_id)
    self._tier1.pop(context_id, None)
    if self._current_context == context_id:
        self._current_context = None
```

The `elif` branch is the only change from the existing implementation. On failure, `negative_merge` is called before clearing Tier 1.

### 3.5 Cap Enforcement

`max_tier2_negative: int = 100` is enforced inside `negative_merge` as a pre-promotion guard:

```python
if len(self._tier2_negative.query_all()) < max_tier2_negative:
    self._tier2_negative.save(p1, w1 * 0.5, aid1)
```

When the cap is reached, new conflict patterns are silently dropped. This is intentional: the negative store is a bounded inhibitory memory, not an unbounded log. A future eviction policy (e.g. drop lowest-weight negative patterns) is deferred (YAGNI).

### 3.6 Introspection Methods

For monitoring and testing:

```python
def query_tier2_negative_all(self) -> list:
    """Return all negative Tier 2 records (for monitoring/testing)."""
    return self._tier2_negative.query_all()
```

---

## 4. Section 2: PatternField Inhibitory Channel

### 4.1 New PatternField Attributes

```python
# In PatternField.__init__:
self._negative: dict[str, list[tuple]] = {}
# Maps agent_id -> [(pattern, weight), ...]
```

This is a parallel channel to `self._agent_patterns` (the positive frequency registry). Unlike the positive channel, the negative channel stores full `(pattern, weight)` tuples rather than just `{pattern_id: weight}`, because inhibitory patterns are pulled and saved directly into the receiving agent's store — UUID sharing is not the mechanism here.

### 4.2 New Method: `broadcast_negative`

```python
def broadcast_negative(self, pattern, weight: float, agent_id: str) -> None:
    """Register a negative pattern from agent_id into the inhibitory channel."""
    if agent_id not in self._negative:
        self._negative[agent_id] = []
    self._negative[agent_id].append((pattern, weight))
```

Called by `Agent.step()` after pulling its own `_tier2_negative` patterns.

### 4.3 New Method: `pull_negative`

```python
def pull_negative(self, agent_id: str, gamma_neg: float) -> list[tuple]:
    """
    Return all negative patterns from other agents, attenuated by gamma_neg.
    Returns list[(pattern, attenuated_weight)].
    """
    result = []
    for src_id, records in self._negative.items():
        if src_id == agent_id:
            continue  # do not pull own patterns back
        for pattern, weight in records:
            result.append((pattern, weight * gamma_neg))
    return result
```

`gamma_neg` is the social inhibition attenuation factor. It plays the same role as `gamma_soc` for positive social learning, but controls how strongly an agent adopts another agent's taboos.

### 4.4 AgentConfig New Fields

Three new fields added to `AgentConfig` with defaults that preserve backward compatibility:

```python
gamma_neg: float = 0.3        # social inhibition attenuation (0 = off)
conflict_threshold: float = 0.7  # cosine sim threshold for negative_merge
max_tier2_negative: int = 100    # cap on _tier2_negative store size
```

Note: `conflict_threshold` already exists in `AgentConfig` (currently used for `RecombinationOperator` trigger logic). **The negative_merge `conflict_threshold` is a separate concept** and should be a distinct parameter name, e.g. `neg_conflict_threshold: float = 0.7`, to avoid shadowing the existing recombination trigger threshold.

### 4.5 Agent.step() Changes

Two additions to `Agent.step()`, executed after the existing field registration:

**Step A — Pull from inhibitory field:**
```python
if self.field is not None:
    neg_incoming = self.field.pull_negative(self.agent_id, self.config.gamma_neg)
    for pattern, weight in neg_incoming:
        self.store._tier2_negative.save(pattern, weight, self.agent_id)
```

This populates the local `_tier2_negative` with socially-learned taboos from peer agents. The save is direct to `_tier2_negative` (bypassing the context-aware `save()` routing, which would incorrectly route to Tier 1 during an active context).

Note: This requires that `self.store` is a `TieredStore`. For backward compatibility with agents using `InMemoryStore`, the pull should be guarded:

```python
if self.field is not None and hasattr(self.store, '_tier2_negative'):
    neg_incoming = self.field.pull_negative(self.agent_id, self.config.gamma_neg)
    for pattern, weight in neg_incoming:
        self.store._tier2_negative.save(pattern, weight, self.agent_id)
```

**Step B — Broadcast own negative patterns:**
```python
if self.field is not None and hasattr(self.store, 'query_negative'):
    for pattern, weight in self.store.query_negative(self.agent_id):
        self.field.broadcast_negative(pattern, weight, self.agent_id)
```

Order: Step A (pull) happens before Step B (broadcast). This prevents an agent from seeing its own patterns reflected back in the same step.

Step A and B are added at the end of `step()`, after the existing `self.field.register(...)` call, to keep the positive and negative channels' update timing consistent.

---

## 5. Section 3: Scoring — ensemble_score (ARC)

### 5.1 Sign Convention

`GaussianPattern.log_prob(x)` returns **Negative Log-Likelihood (NLL)** — a positive scalar where lower values indicate higher probability under the pattern. This is confirmed by the docstring in `benchmarks/multi_agent_arc.py`:

> Sign convention: GaussianPattern.log_prob returns NLL (lower = more probable).

And the comment in `Agent._accept_communicated()`:

> Sign convention: log_prob(x) returns NLL (positive). -log_prob(x) gives log-likelihood (<= 0).

### 5.2 Updated ensemble_score

```python
def ensemble_score(agents, vec: np.ndarray) -> float:
    """
    Compute ensemble score for a candidate vector.

    Positive patterns contribute positively (higher NLL = less probable = worse candidate).
    Negative patterns are subtracted (lower NLL for negative pattern = candidate resembles
    taboo = penalise by reducing the positive contribution).

    total += pos_weights * pos_NLL   (existing behaviour)
    total -= neg_weights * neg_NLL   (new inhibitory term)

    Lower total = more preferred candidate (consistent with existing ranking logic).
    Returns 0.0 if all stores are empty.
    """
    total = 0.0
    any_records = False
    for agent in agents:
        pos = agent.store.query(agent.agent_id)
        if pos:
            any_records = True
            total += sum(w * p.log_prob(vec) for p, w in pos)
        neg = agent.store.query_negative(agent.agent_id)
        if neg:
            any_records = True
            total -= sum(w * p.log_prob(vec) for p, w in neg)
    return total if any_records else 0.0
```

### 5.3 Correctness of Sign Logic

The existing `evaluate_task` ranks candidates by `ensemble_score`: lower score = more probable = better candidate. The correct answer wins if `correct_score < min(distractor_scores)`.

With the inhibitory term, consider a candidate `x` that strongly resembles a negative pattern (small `neg_NLL`):

- `neg_NLL` is small → `-(w * neg_NLL)` is a small negative number
- `total` is *less reduced* than for a candidate far from the negative pattern

Wait — this is backwards. Let us re-examine carefully:

- `x` close to negative pattern μ → NLL is **low** → `w * NLL` is small → `total -= small` → total is less affected → score stays higher (worse)
- `x` far from negative pattern μ → NLL is **high** → `w * NLL` is large → `total -= large` → total is lower → score is better

So: candidates close to taboo patterns retain a higher score (are penalised, ranked worse). Candidates far from taboo patterns get their score reduced (ranked better). This is the correct inhibitory direction: the correct answer, which should be *different* from failed-task patterns, benefits from the inhibitory term relative to distractors that happen to resemble taboo patterns.

**Example walkthrough:**
- Correct output: `correct_score = 50 + 5 * NLL_neg` (moderately far from negative pattern, NLL_neg = 10 → `-= 10`, net 40)
- Distractor: `distractor_score = 48 + 5 * NLL_neg_distractor` (close to negative pattern, NLL_neg = 2 → `-= 2`, net 46)
- Result: `correct_score (40) < distractor_score (46)` → correct wins

The inhibitory term converts partial confusions into resolved discriminations.

### 5.4 Backward Compatibility

`query_negative` returns `[]` when `_tier2_negative` is empty (which it always is at the start and on fresh orchestrators). The `neg` branch is only entered when records exist. This means the updated `ensemble_score` is strictly additive: with no negative patterns, scores are identical to the current implementation.

---

## 6. Section 4: Lifecycle and Monitoring

### 6.1 Full Context Lifecycle

```
begin_context(context_id)
    └─ Fresh Tier 1 store created

[Training steps: Agent.step() writes to Tier 1 via TieredStore.save()]

end_context(context_id, correct=True)
    └─ similarity_merge(context_id)  →  Tier 2 positive  [EXISTING]
    └─ _tier1[context_id] cleared

end_context(context_id, correct=False)
    └─ negative_merge(context_id)    →  _tier2_negative  [NEW]
    └─ _tier1[context_id] cleared
```

No change to the successful path. The failed path now routes through `negative_merge` instead of immediately discarding.

### 6.2 CrossTaskRecombinator: Unchanged

The `CrossTaskRecombinator.consolidate()` method operates on `TieredStore.query_tier2_all()` (positive Tier 2 only). It is **not modified** to handle negative patterns. Reasons:
- Recombining negative patterns to produce new negative patterns has unclear semantics.
- The inhibitory store is a boundary/taboo mechanism, not a generative one.
- YAGNI: introduce only if empirical evidence shows recombination of taboo patterns helps.

### 6.3 StructuralLawMonitor: New Metrics

The `StructuralLawMonitor` (tracked via `T_monitor` interval in `run_persistent`) gains two new diagnostic metrics:

**`negative_count`**: Total number of patterns across all agents' `_tier2_negative` stores.

```python
negative_count = sum(
    len(agent.store.query_negative(agent.agent_id))
    for agent in agents
    if hasattr(agent.store, 'query_negative')
)
```

This tracks growth of the inhibitory memory over time. If `negative_count` grows unboundedly, the cap (`max_tier2_negative`) is being hit and patterns are being silently dropped — an operational alert.

**`taboo_overlap`**: Fraction of negative pattern UUIDs shared across all agents.

```python
def taboo_overlap(agents) -> float:
    """Fraction of negative pattern UUIDs present in ALL agents vs any agent."""
    neg_id_sets = [
        {p.id for p, _ in agent.store.query_negative(agent.agent_id)}
        for agent in agents
        if hasattr(agent.store, 'query_negative')
    ]
    if not neg_id_sets or not any(neg_id_sets):
        return 0.0
    intersection = neg_id_sets[0].intersection(*neg_id_sets[1:])
    union = neg_id_sets[0].union(*neg_id_sets[1:])
    return len(intersection) / len(union) if union else 0.0
```

`taboo_overlap` measures social consensus on taboo patterns. High overlap means agents have converged on shared inhibitory knowledge (positive). Very high overlap (> threshold) may indicate all agents are over-constrained by the same taboos ("Fear" state).

### 6.4 RecombinationStrategist: Fear Reset

The `RecombinationStrategist` (which currently manages diversity-collapse intervention) gains a new intervention mode: **Fear Reset**.

**Trigger condition:** `taboo_overlap > fear_threshold` (default `fear_threshold = 0.8`).

**Intervention:** Temporarily set `gamma_neg = 0.0` for all agents for `fear_reset_duration` steps (default 20 steps).

**Mechanism:** By zeroing `gamma_neg`, agents stop pulling negative patterns from the field. Each agent then learns from its own task experience without being influenced by the shared taboo pool. If some agents encounter tasks where the taboo pattern is actually predictive, they can re-learn it from scratch, breaking the collective over-inhibition.

**Implementation sketch:**

```python
class RecombinationStrategist:
    # existing fields ...
    fear_threshold: float = 0.8
    fear_reset_duration: int = 20
    _fear_reset_remaining: int = 0

    def check(self, monitor_metrics: dict, agents: list) -> None:
        # existing diversity/redundancy checks ...

        taboo = monitor_metrics.get('taboo_overlap', 0.0)
        if taboo > self.fear_threshold and self._fear_reset_remaining == 0:
            self._fear_reset_remaining = self.fear_reset_duration
            for agent in agents:
                agent._saved_gamma_neg = agent.config.gamma_neg
                agent.config.gamma_neg = 0.0

        if self._fear_reset_remaining > 0:
            self._fear_reset_remaining -= 1
            if self._fear_reset_remaining == 0:
                for agent in agents:
                    agent.config.gamma_neg = getattr(agent, '_saved_gamma_neg', 0.3)
```

This is the only change to `RecombinationStrategist`. The existing diversity/redundancy logic is not modified.

---

## 7. Testing Approach

### 7.1 Unit Tests: TieredStore negative_merge

File: `tests/store/test_tiered_store_negative.py`

| Test | Description |
|------|-------------|
| `test_negative_merge_promotes_conflicting` | Create a positive Tier 2 pattern with mu=[1,0,...]. Add a Tier 1 pattern from a failed task with mu=[0.95,0,...] (cosine sim ≈ 0.95 ≥ 0.7). Call `end_context(correct=False)`. Assert `query_negative` returns 1 pattern. |
| `test_negative_merge_discards_non_conflicting` | Add a Tier 1 pattern with mu orthogonal to all Tier 2 positives (cosine sim = 0). Call `end_context(correct=False)`. Assert `query_negative` returns 0 patterns. |
| `test_negative_merge_respects_cap` | Fill `_tier2_negative` to capacity (100). Add a conflicting Tier 1 pattern. Assert `query_negative` still returns exactly 100 (new pattern silently dropped). |
| `test_end_context_correct_no_negative_merge` | Call `end_context(correct=True)`. Assert `query_negative` is empty (negative_merge not called on success). |
| `test_query_negative_empty_by_default` | Fresh `TieredStore`. Assert `query_negative('any_agent')` returns `[]`. |

### 7.2 Unit Tests: PatternField inhibitory channel

File: `tests/field/test_field_negative.py`

| Test | Description |
|------|-------------|
| `test_broadcast_negative_stores_pattern` | Call `broadcast_negative(pattern, 1.0, 'agent_a')`. Assert `pull_negative('agent_b', 1.0)` returns that pattern. |
| `test_pull_negative_excludes_own` | `broadcast_negative(pattern, 1.0, 'agent_a')`. Assert `pull_negative('agent_a', 1.0)` returns `[]`. |
| `test_pull_negative_attenuates_weight` | Broadcast with weight 1.0, pull with gamma_neg=0.3. Assert returned weight is 0.3. |
| `test_multiple_agents_broadcast` | Agents A and B broadcast different negative patterns. Agent C pulls both, attenuated. |

### 7.3 Unit Tests: ensemble_score inhibitory subtraction

File: `tests/benchmarks/test_arc_ensemble_score_negative.py`

| Test | Description |
|------|-------------|
| `test_inhibitory_raises_score_for_close_candidate` | Mock agent with one negative pattern centered on distractor. Assert distractor score is *higher* (worse) than without inhibitory term. |
| `test_inhibitory_no_effect_when_empty` | No negative patterns. Assert ensemble_score output identical to current behaviour. |
| `test_correct_wins_with_inhibitory` | Set up a scenario where correct candidate is far from negative patterns, distractor is close. Assert correct_score < distractor_score. |

### 7.4 Integration Test: ARC persistent run with inhibitory tier

File: `tests/benchmarks/test_arc_persistent_negative.py`

Test: Run `run_persistent` on 20 ARC tasks with negative patterns enabled. After run:
- Assert `tiered.query_tier2_negative_all()` has length > 0 (some failures promoted patterns).
- Assert final accuracy >= accuracy from run without inhibitory tier (or at worst within 2 percentage points, to allow for variance).
- Assert `taboo_overlap` is computable and in [0.0, 1.0].

### 7.5 Regression: Existing tests must pass unchanged

All existing tests in `tests/store/test_tiered_store.py` must pass without modification. The new `_tier2_negative` partition is completely isolated from existing methods (`query`, `save`, `load`, `update_weight`, `similarity_merge`, `promote_to_tier2`).

---

## 8. Open Questions

### 8.1 Eviction Policy for _tier2_negative

When `max_tier2_negative` is reached, new conflict patterns are silently dropped (FIFO by cap). An alternative is LRU (drop lowest-weight pattern). Current approach is FIFO-like (drop on overflow). **Decision deferred** — collect empirical data on how often the cap is hit before investing in eviction logic.

### 8.2 Weight Decay for Negative Patterns

Positive Tier 2 patterns get weight *boosts* when similar patterns recur across successful tasks (via `similarity_merge`). Negative patterns have no symmetric decay mechanism — a taboo pattern from a failed task retains its weight indefinitely. Over very long runs, this could lead to stale inhibitory patterns blocking otherwise-useful structural regions.

**Proposed future work:** Add a `negative_decay` multiplier applied each time `negative_merge` is called, so taboo weights decay unless refreshed by recurring failures. Not implemented in this design.

### 8.3 Pattern Identity in Negative Store

`query_negative` returns patterns by agent_id, using `InMemoryStore.query()`. If two agents independently learn similar (but distinct UUID) negative patterns, `taboo_overlap` will not detect the overlap because it matches by UUID identity, not cosine similarity.

A cosine-similarity-based `taboo_overlap` would be more semantically accurate but is more expensive. The UUID-based version is a reasonable approximation for agents that share patterns via `broadcast_negative` (which creates new UUIDs on copy). **Deferred** — evaluate after seeing real taboo_overlap data.

### 8.4 Interaction with RecombinationOperator

The `RecombinationOperator` uses `conflict_threshold` from `AgentConfig` to determine recombination eligibility. Section 3.4 notes that the `neg_conflict_threshold` for `negative_merge` should be a *distinct* config field. The spec uses `neg_conflict_threshold: float = 0.7` as the name. **Implementation must confirm there is no field-name collision** with the existing `conflict_threshold: float = 0.1` in `AgentConfig` (used for recombination trigger logic, a different concept).

### 8.5 Fear Reset and Asymmetric Agent Behaviour

During Fear Reset, `gamma_neg` is zeroed for all agents uniformly. If agents have different taboo histories (different `_tier2_negative` contents), zeroing gamma_neg has different effects per agent. An agent with many false taboos benefits more from Fear Reset than one with few. Whether uniform Fear Reset is the right mechanism — vs. per-agent assessment — is an open question for empirical evaluation.

---

## 9. Summary of Interface Changes

### New/modified public methods

| Class | Method | Status |
|-------|--------|--------|
| `TieredStore` | `query_negative(agent_id)` | New |
| `TieredStore` | `negative_merge(context_id, conflict_threshold, max_tier2_negative)` | New |
| `TieredStore` | `query_tier2_negative_all()` | New |
| `TieredStore` | `end_context(context_id, correct)` | Modified (elif branch) |
| `PatternField` | `broadcast_negative(pattern, weight, agent_id)` | New |
| `PatternField` | `pull_negative(agent_id, gamma_neg)` | New |
| `AgentConfig` | `gamma_neg`, `neg_conflict_threshold`, `max_tier2_negative` | New fields |
| `Agent` | `step()` | Modified (pull/broadcast negative at end) |
| `ensemble_score` (ARC) | N/A | Modified (subtract neg NLL) |
| `StructuralLawMonitor` | `negative_count`, `taboo_overlap` | New metrics |
| `RecombinationStrategist` | `check()` | Modified (Fear Reset branch) |

### No changes to

- `GaussianPattern` (polarity by partition, not by field)
- `InMemoryStore` (used as underlying store for `_tier2_negative`)
- `CrossTaskRecombinator`
- `similarity_merge` (existing positive merge logic)
- `PatternStore` protocol (no new required methods)
- `EpistemicEvaluator`, `AffectiveEvaluator`, `SocialEvaluator`, `ResourceCostEvaluator`
- `MetaPatternRule`, `RecombinationOperator`
