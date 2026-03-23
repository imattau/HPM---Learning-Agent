# Task Pattern Library — Sub-project 3 Design Spec

## Overview

This spec covers Sub-project 3 of the hierarchical extension to the HPM Learning Agent. It extends `ContextualPatternStore` to archive and retrieve **task-level patterns** represented as L3 bundles — enabling conceptual warm-starting across tasks, not just sensory warm-starting from substrate structure.

Sub-projects:
- **Sub-project 1 (done):** Inter-level encoding protocol + `HierarchicalOrchestrator` (2-level stack)
- **Sub-project 2 (done):** N-level `StackedOrchestrator` + hierarchical ARC benchmark
- **Sub-project 3 (this spec):** Task patterns via L3 bundle archiving + conceptual retrieval

---

## Motivation

The current `ContextualPatternStore` uses hand-crafted grid structural features (`SubstrateSignature`: grid size, colour count, aspect ratio) for coarse filtering, then ranks candidates by NLL of the first few encoded training vectors. This is *sensory retrieval* — it finds tasks that look similar at the substrate level.

ARC tasks are deliberately diverse in substrate appearance but share transformation rules (rotation, symmetry, recolouring, gravity, etc.). Two tasks may share the same logic but look completely different in 64-dim projection space. Sensory retrieval cannot recognise this equivalence.

The benchmark investigation (Sub-project 2) showed only 1 Tier 2 promotion across 342 tasks: the 64-dim random projection collapses all ARC tasks toward a near-universal direction, preventing the TieredStore from building a meaningful cross-task pattern library.

The fix: represent each task as a **L3 bundle** — the 68-dim abstraction formed after training. L3 filters out substrate noise and encodes the transformation rule. Archiving L3 bundles from successful tasks and using them for retrieval shifts the system from sensory to **conceptual retrieval**.

---

## Design Decisions

### L3 bundle as the task pattern

The L3 bundle extracted after training (68-dim: mu concatenated with [weight, epistemic_loss]) is the natural task pattern:

- **Conceptual invariance**: L3 captures relational structure via L2 bundles derived from L1 features, filtering out pixel-level variation
- **Epistemic weighting**: the bundle includes weight and epistemic_loss, encoding the agent's certainty about the learned rule
- **Dimension integrity**: treating the task as a high-level pattern allows the Librarian to retrieve past tasks by conceptual neighbourhood in 68-dim logic space

### Gate on success only

L3 bundles are archived only when `correct=True`. This keeps the archive as validated intelligence — failed hypotheses are discarded. The archive is a semantic memory of transformation rules that actually solved problems.

### Two-stage retrieval at begin_context

1. **Coarse filter** — `SubstrateSignature` (grid size, colour count, aspect ratio): unchanged, existing `ArchiveLibrarian.query_archive()`
2. **Fine ranking** — L3 bundle NLL: rank surviving candidates by NLL of the current L3 state against stored L3 bundles. `ArchiveForecaster.rank()` is called with the current L3 bundle encoded as a vector (instead of `first_obs` ARC pair deltas). Falls back to `first_obs` NLL when L3 agents are not provided or no L3 bundles exist in the archive (cold start, task 0).

### L3 warm-start injection

After retrieving the best-matching archive, the stored L3 bundles are injected as seed patterns into L3 agents' stores via `agent.store.save(pattern, weight, agent_id)`. L3 agents begin each task with a biased prior toward the transformation logic that solved the most similar past task.

### Backward compatibility

`l3_agents=None` at construction preserves existing behaviour exactly — no bundle extraction, no L3 injection, fallback to `first_obs` NLL ranking. `multi_agent_arc.py` and all existing consumers are unchanged.

Archive files in old list format remain readable — `_load_archive` checks `isinstance(records, list)` before accessing `records["tier2"]`.

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

Old format (list of tuples) remains readable. `_load_archive` handles both:
- `isinstance(records, list)` → old format: load tier2 only, return empty l3_bundles
- `isinstance(records, dict)` → new format: load both tier2 and l3_bundles

**`end_context` change:**

If `correct=True` and `l3_agents` provided, extract bundles and write dict-format archive:
```python
{
    "tier2": tier2_state,
    "l3_bundles": [(bundle.mu.copy(), float(bundle.weight), float(bundle.epistemic_loss))
                   for agent in self._l3_agents
                   for bundle in [extract_bundle(agent)]]
}
```

**`begin_context` change:**

After coarse filter retrieval, `_load_archive` returns l3_bundles in addition to loading Tier 2. A new `_inject_l3(l3_bundles)` helper injects each stored bundle as a GaussianPattern seed into all L3 agents' stores.

For ranking: if `l3_agents` provided and agents have non-seed patterns, encode current L3 state and pass to `ArchiveForecaster.rank()` as `first_obs`. Otherwise falls back to caller-supplied `first_obs` (existing path).

**New helper `_inject_l3`:**
```python
def _inject_l3(self, l3_bundles: list) -> None:
    if not self._l3_agents or not l3_bundles:
        return
    for mu, weight, epistemic_loss in l3_bundles:
        pattern = make_pattern(mu=mu, scale=np.eye(len(mu)), pattern_type="gaussian")
        for agent in self._l3_agents:
            agent.store.save(pattern, weight, agent.agent_id)
```

### Modified file: `benchmarks/hierarchical_arc.py`

One change in `run_persistent()` — pass `l3_agents` to hierarchical store:
```python
l1_contextual = ContextualPatternStore(
    tiered_store=l1_tiered,
    archive_dir="data/archives/hierarchical_arc_persistent",
    l3_agents=stacked_orch.level_agents[-1],
)
```

Flat baseline keeps `l3_agents=None`.

### Unchanged files

- `hpm/store/archive_librarian.py` — SubstrateSignature coarse filter unchanged
- `hpm/store/archive_forecaster.py` — NLL ranking logic unchanged; called with different `first_obs`
- `hpm/store/tiered_store.py`, `hpm/agents/agent.py` — untouched
- `benchmarks/multi_agent_arc.py` — unchanged (no `l3_agents` passed)

### New tests: `tests/store/test_contextual_store.py`

- **L3 round-trip**: archive on success with `l3_agents` → dict format with `l3_bundles` key → reload and verify mu arrays match
- **Old format compat**: old list-format archive still loads without error; l3_bundles returns empty list
- **L3 injection**: after `begin_context` with a matching archive containing L3 bundles, L3 agent store contains the seeded pattern
- **Fallback (no l3_agents)**: with `l3_agents=None`, behaviour identical to current

---

## Data flow summary

```
Task N training complete, correct=True
    │
    ▼
end_context():
    extract_bundle(l3_agent_0) -> (mu_0, w_0, eps_0)   # 68-dim
    write archive: {tier2: [...], l3_bundles: [(mu_0, w_0, eps_0), ...]}
    write index.json entry

Task N+1 begins -> begin_context(sig, first_obs_arg):
    ArchiveLibrarian.query_archive(sig) -> candidates (coarse filter)
    encode current L3 state -> first_obs for ranking
    ArchiveForecaster.rank(candidates, first_obs) -> ranked
    _load_archive(best.archive_path) -> loads Tier 2 + returns l3_bundles
    _inject_l3(l3_bundles) -> l3_agent.store.save(pattern, w, agent_id)
    self._store.begin_context(context_id) -> Tier 1 open
```

---

## What is NOT in this spec

- Negative L3 pattern archiving (taboo rules from failed tasks)
- Confidence-gated stepping (epistemic routing to skip levels when uncertain)
- Top-down prediction from L3 to L1
- Multiple L3 bundle injection (top-K matches, not just best-1)

---

## Success criteria

- `ContextualPatternStore` with `l3_agents` archives L3 bundles on success and injects them at `begin_context`; all existing tests still pass
- `hierarchical_arc.py --persistent` shows measurably more cross-task pattern building than the 1 Tier 2 promotion seen with bare TieredStore
- Flat baseline (no `l3_agents`) is unaffected and serves as comparison
- New tests cover round-trip, backward compat, injection, and fallback
