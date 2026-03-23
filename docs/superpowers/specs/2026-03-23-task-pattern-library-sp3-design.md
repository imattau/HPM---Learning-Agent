# Task Pattern Library — Sub-project 3 Design Spec

## Overview

This spec covers Sub-project 3 of the hierarchical extension to the HPM Learning Agent. It extends `ContextualPatternStore` to archive and retrieve **task-level patterns** represented as L3 bundles — enabling conceptual warm-starting across tasks, not just sensory warm-starting from substrate structure.

Sub-projects:
- **Sub-project 1 (done):** Inter-level encoding protocol + `HierarchicalOrchestrator`
- **Sub-project 2 (done):** N-level `StackedOrchestrator` + hierarchical ARC benchmark
- **Sub-project 3 (this spec):** Task patterns via L3 bundle archiving + conceptual retrieval

---

## Motivation

The current `ContextualPatternStore` uses hand-crafted grid structural features (`SubstrateSignature`) for coarse filtering, then ranks candidates by NLL of the first few encoded training vectors against stored Tier 2 GaussianPatterns. This is *sensory retrieval*.

ARC tasks are deliberately diverse in substrate appearance but share transformation rules (rotation, symmetry, recolouring, gravity, etc.). The Sub-project 2 benchmark showed only 1 Tier 2 promotion across 342 tasks: the 64-dim random projection collapses all ARC tasks toward a near-universal direction, so sensory retrieval cannot distinguish between tasks.

The fix: represent each task as a **L3 bundle** (68-dim) extracted after training. L3 filters out substrate noise and encodes the transformation rule. Archiving L3 bundles from successful tasks and re-ranking by L3 cosine similarity shifts the system from sensory to **conceptual retrieval**.

---

## Design Decisions

### L3 bundle as the task pattern

The L3 bundle extracted after training (68-dim: mu concatenated with [weight, epistemic_loss]) is the natural task pattern:

- **Conceptual invariance**: L3 captures relational structure via L2 bundles derived from L1 features, filtering out pixel-level variation
- **Epistemic weighting**: `weight` in the bundle encodes the agent's confidence in the learned rule; `epistemic_loss` encodes residual uncertainty
- **Dimension integrity**: treating the task as a high-level pattern allows retrieval by conceptual neighbourhood in 68-dim logic space

### Gate on success only

L3 bundles are archived only when `correct=True`. Failed hypotheses are discarded. The archive is a semantic memory of transformation rules that actually solved problems.

### Two-stage retrieval at begin_context

1. **Coarse filter** — `SubstrateSignature` (grid size, colour count, aspect ratio): unchanged, existing `ArchiveLibrarian.query_archive()`
2. **Fine re-ranking** — L3 cosine similarity: rank surviving candidates by cosine similarity between the current L3 bundle and each candidate's stored L3 bundle arrays. This is handled directly inside `ContextualPatternStore._rank_by_l3()` — **not** via `ArchiveForecaster.rank()` (which operates in 64-dim Tier 2 space and cannot score 68-dim L3 vectors). Falls back to `ArchiveForecaster.rank()` (existing first_obs NLL path) when L3 agents are not provided or no candidates have stored L3 bundles.

### L3 warm-start injection

After retrieving the best-matching archive, stored L3 bundle arrays are injected as GaussianPattern seeds into L3 agents' stores. Each bundle is reconstructed with `mu=stored_mu`, `scale=np.eye(68)` (identity covariance). The identity covariance is a deliberate choice: the `weight` argument to `store.save()` already encodes the bundle's certainty (high-weight patterns dominate ensemble scoring); using identity covariance keeps the injected prior broad, avoiding over-constraining the L3 agent before it has seen the new task's training data.

### Backward compatibility

`l3_agents=None` at construction preserves existing behaviour exactly. `multi_agent_arc.py` and all existing consumers are unchanged. Old list-format archive files remain readable.

---

## Architecture

### Modified file: `hpm/store/contextual_store.py`

**`__init__` signature change:**
```python
def __init__(
    self,
    tiered_store,
    archive_dir: str,
    run_id: Optional[str] = None,
    fingerprint_nll_threshold: float = 50.0,
    global_weight_threshold: float = 0.6,
    global_promotion_n: int = 5,
    l3_agents: list | None = None,   # new: L3 agent list for bundle archiving/injection
):
```

**Archive format change:**

New format (dict):
```python
{
    "tier2": [(pattern, weight, agent_id), ...],
    "l3_bundles": [(mu_array, weight, epistemic_loss), ...],  # numpy arrays, one per L3 agent
}
```

Old format (plain list of 3-tuples) remains readable via `isinstance(records, list)` check.

**`_load_archive` signature change:**
```python
def _load_archive(self, archive_path: str) -> list:
    """Load archive from disk. Returns l3_bundles list (empty list for old-format files).

    Side effect: replaces Tier 2 contents with archived patterns.
    """
```

Old format: load tier2 only, return `[]`.
New format: load tier2 and return `records["l3_bundles"]`.

**`end_context` change:**

After the existing Tier 2 archive write, if `correct=True` and `self._l3_agents`:
```python
from hpm.agents.hierarchical import extract_bundle
l3_bundles = []
for agent in self._l3_agents:
    b = extract_bundle(agent)
    l3_bundles.append((b.mu.copy(), float(b.weight), float(b.epistemic_loss)))
payload = {"tier2": tier2_state, "l3_bundles": l3_bundles}
# write payload to tmp_path, then os.replace to archive_path
```

**New method `_rank_by_l3`:**
```python
def _rank_by_l3(self, candidates: list, current_l3_vecs: list[np.ndarray]) -> list:
    """Re-rank candidates by cosine similarity of their stored L3 bundles to current L3 state.

    current_l3_vecs: list of encoded L3 bundle vectors (one per L3 agent).
    Candidates without stored L3 bundles are moved to the end of the ranking.
    Returns re-ranked candidates list.
    """
```

Similarity: for each candidate, load its L3 bundles from the archive index (not a full pkl load — index.json stores the archive path, load just the l3_bundles key). Score = mean cosine similarity between each stored l3 bundle and each current_l3_vec.

**`begin_context` change:**

After `ArchiveLibrarian.query_archive(sig)` returns candidates:
1. If `self._l3_agents`: encode current L3 state as `[encode_bundle(extract_bundle(a)) for a in self._l3_agents]`, call `_rank_by_l3(candidates, current_l3_vecs)` to re-rank
2. Else: call `self._forecaster.rank(candidates, first_obs, ...)` as before
3. `l3_bundles = self._load_archive(best.archive_path)` — returns list (may be empty)
4. `self._inject_l3(l3_bundles)` — injects into L3 agents' stores
5. `self._store.begin_context(context_id)` — opens Tier 1

**New method `_inject_l3`:**
```python
def _inject_l3(self, l3_bundles: list) -> None:
    """Inject stored L3 bundle arrays as GaussianPattern seeds into L3 agents' stores."""
    if not self._l3_agents or not l3_bundles:
        return
    from hpm.patterns.factory import make_pattern
    for mu, weight, epistemic_loss in l3_bundles:
        pattern = make_pattern(mu=mu, scale=np.eye(len(mu)), pattern_type="gaussian")
        for agent in self._l3_agents:
            agent.store.save(pattern, weight, agent.agent_id)
```

### Modified file: `benchmarks/hierarchical_arc.py`

One change in `run_persistent()`:
```python
l1_contextual = ContextualPatternStore(
    tiered_store=l1_tiered,
    archive_dir="data/archives/hierarchical_arc_persistent",
    l3_agents=stacked_orch.level_agents[-1],   # L3 agents for conceptual retrieval
)
```

Flat baseline keeps `l3_agents=None`.

### Unchanged files

- `hpm/store/archive_librarian.py` — SubstrateSignature coarse filter unchanged
- `hpm/store/archive_forecaster.py` — NLL ranking unchanged; still used when `l3_agents=None`
- `hpm/store/tiered_store.py`, `hpm/agents/agent.py` — untouched
- `benchmarks/multi_agent_arc.py` — unchanged

### New tests: `tests/store/test_contextual_store.py`

- **L3 round-trip**: archive on success with `l3_agents` → pkl is dict with `l3_bundles` key → `_load_archive` returns list of (mu_array, weight, eps) tuples; mu arrays match originals
- **Old format compat**: old list-format archive loads without error; `_load_archive` returns `[]`
- **L3 injection**: after `begin_context` with a matching archive containing L3 bundles, L3 agent store contains ≥1 seeded pattern with matching mu
- **Fallback (no l3_agents)**: with `l3_agents=None`, behaviour identical to current — `_rank_by_l3` not called, no injection

---

## Data flow summary

```
Task N training complete, correct=True
    │
    ▼
end_context():
    extract_bundle(l3_agent_0) -> LevelBundle(mu_0, w_0, eps_0)
    write archive dict: {tier2: [...], l3_bundles: [(mu_0, w_0, eps_0)]}
    update index.json

Task N+1 begins -> begin_context(sig, first_obs_arg):
    ArchiveLibrarian.query_archive(sig) -> candidates (coarse, substrate filter)
    if l3_agents:
        current_l3_vecs = [encode_bundle(extract_bundle(a)) for a in l3_agents]
        candidates = _rank_by_l3(candidates, current_l3_vecs)   # cosine sim re-rank (68-dim)
    else:
        candidates = forecaster.rank(candidates, first_obs_arg)   # NLL re-rank (64-dim, existing)
    l3_bundles = _load_archive(best.archive_path)  # returns list[tuple], loads Tier 2 as side effect
    _inject_l3(l3_bundles)                          # seeds L3 agents (no-op if empty)
    self._store.begin_context(context_id)            # opens Tier 1
```

---

## What is NOT in this spec

- Negative L3 pattern archiving (taboo rules from failed tasks)
- Confidence-gated stepping (epistemic routing)
- Top-down prediction from L3 to L1
- Multiple L3 bundle injection (top-K matches, not just best-1)
- Separate `rank_l3()` method on `ArchiveForecaster` (L3 ranking stays in ContextualPatternStore)

---

## Success criteria

- `ContextualPatternStore` with `l3_agents` archives L3 bundles on success and injects them at next `begin_context`; all 651 existing tests still pass
- After running `hierarchical_arc.py --persistent` on 342 tasks, L3 archive contains at least 10 entries (i.e., at least 10 tasks solved correctly, with L3 bundles saved)
- A unit test confirms that after `begin_context` with a matching L3 archive, the L3 agent store contains at least one injected pattern whose `mu` matches the archived bundle
- Flat baseline (no `l3_agents`) is unaffected and serves as comparison
