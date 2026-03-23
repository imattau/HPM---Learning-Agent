# Task Pattern Library (Sub-project 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `ContextualPatternStore` to archive L3 bundles from successful tasks and use them for conceptual warm-starting — replacing sensory NLL ranking with cosine similarity in L3 abstraction space.

**Architecture:** Two changes to `hpm/store/contextual_store.py`: (1) archive format extended from a list to a dict containing both Tier 2 patterns and L3 bundle arrays; (2) optional `l3_agents` parameter enables `_rank_by_l3()` (cosine sim in L3 space) and `_inject_l3()` (seed L3 stores at begin_context). The benchmark restructures `run_persistent()` to build `l1_contextual` after `stacked_orch` so that `l3_agents` are available at construction time.

**Tech Stack:** Python, numpy, pickle (existing), hpm.agents.hierarchical (extract_bundle, encode_bundle), hpm.patterns.factory (make_pattern)

---

## File Structure

| File | Change |
|---|---|
| `hpm/store/contextual_store.py` | Add `l3_agents` param; update `_load_archive` return type; add `_rank_by_l3`, `_inject_l3`; extend `end_context` archive write; update `begin_context` branching |
| `benchmarks/hierarchical_arc.py` | Restructure `run_persistent()`: build `l1_orch` with bare `l1_tiered`, build `stacked_orch`, then build `l1_contextual` with `l3_agents` |
| `tests/store/test_contextual_store.py` | Add 4 new tests (round-trip, compat, injection, fallback) |

---

## Task 1: Update archive format and `_load_archive` return type

Changes `_load_archive` to handle both old (list) and new (dict) archive formats and return `l3_bundles`.

**Files:**
- Modify: `hpm/store/contextual_store.py` (lines 208-215)
- Test: `tests/store/test_contextual_store.py`

Background: The current archive pkl is a plain list of `(pattern, weight, agent_id)` 3-tuples. The new format is a dict `{"tier2": [...], "l3_bundles": [(mu_array, weight, eps), ...]}`. `_load_archive` currently returns `None`; it must return a list of l3_bundle tuples so `begin_context` can inject them.

- [ ] **Step 1: Write the failing tests**

Add to `tests/store/test_contextual_store.py` (note: use `_make_contextual_store` — different name from the existing `_make_store_with_pattern` helper already in the file):

```python
# ---------------------------------------------------------------------------
# Task 1: Archive format round-trip and backward compat
# ---------------------------------------------------------------------------

def _make_contextual_store(archive_dir, dim=8):
    """Helper: ContextualPatternStore with one Tier 2 pattern in a temp dir."""
    tiered = TieredStore()
    store = ContextualPatternStore(tiered_store=tiered, archive_dir=archive_dir)
    p = make_pattern(mu=np.ones(dim), scale=np.eye(dim), pattern_type="gaussian")
    tiered._tier2.save(p, 1.0, "agent_x")
    return store, tiered


def test_load_archive_old_list_format_returns_empty_l3(tmp_path):
    """Old list-format pkl loads without error; _load_archive returns []."""
    import pickle
    store, tiered = _make_contextual_store(str(tmp_path))
    archive_path = str(tmp_path / "old.pkl")
    tier2 = tiered.query_tier2_all()
    with open(archive_path, "wb") as f:
        pickle.dump(tier2, f)
    l3_bundles = store._load_archive(archive_path)
    assert l3_bundles == []
    # Side effect: Tier 2 was loaded from archive
    assert len(tiered.query_tier2_all()) > 0


def test_load_archive_new_dict_format_returns_l3_bundles(tmp_path):
    """New dict-format pkl: _load_archive returns list of (mu, w, eps) tuples."""
    import pickle
    store, tiered = _make_contextual_store(str(tmp_path))
    archive_path = str(tmp_path / "new.pkl")
    tier2 = tiered.query_tier2_all()
    mu = np.array([1.0, 2.0, 3.0])
    payload = {"tier2": tier2, "l3_bundles": [(mu, 0.9, 0.1)]}
    with open(archive_path, "wb") as f:
        pickle.dump(payload, f)
    l3_bundles = store._load_archive(archive_path)
    assert len(l3_bundles) == 1
    stored_mu, stored_w, stored_eps = l3_bundles[0]
    np.testing.assert_array_almost_equal(stored_mu, mu)
    assert stored_w == pytest.approx(0.9)
    assert stored_eps == pytest.approx(0.1)
```

- [ ] **Step 2: Run the tests — verify they fail**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_load_archive_old_list_format_returns_empty_l3 tests/store/test_contextual_store.py::test_load_archive_new_dict_format_returns_l3_bundles -v
```

Expected: FAIL — `_load_archive` currently returns `None`.

- [ ] **Step 3: Update `_load_archive` to return l3_bundles**

In `hpm/store/contextual_store.py`, replace the `_load_archive` method (lines 208-215):

```python
def _load_archive(self, archive_path: str) -> list:
    """Load archive from disk. Replace Tier 2 with archived patterns.

    Returns l3_bundles: list of (mu_array, weight, epistemic_loss) tuples.
    Returns [] for old list-format archives (backward compatible).
    """
    with open(archive_path, "rb") as f:
        records = pickle.load(f)

    if isinstance(records, list):
        # Old format: plain list of (pattern, weight, agent_id)
        tier2_records = records
        l3_bundles = []
    else:
        # New format: dict with "tier2" and "l3_bundles" keys
        tier2_records = records.get("tier2", [])
        l3_bundles = records.get("l3_bundles", [])

    # Replace Tier 2 (clean REPLACE, not additive)
    self._store._tier2._data.clear()
    for pattern, weight, agent_id in tier2_records:
        self._store.promote_to_tier2(pattern, weight, agent_id)

    return l3_bundles
```

Note: a full `pickle.load` is required to read either format — pickle doesn't support partial key loading. At 342 tasks this is acceptable.

- [ ] **Step 4: Run the tests — verify they pass**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_load_archive_old_list_format_returns_empty_l3 tests/store/test_contextual_store.py::test_load_archive_new_dict_format_returns_l3_bundles -v
```

Expected: PASS

- [ ] **Step 5: Verify existing tests still pass**

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

Expected: all existing tests PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add hpm/store/contextual_store.py tests/store/test_contextual_store.py
git commit -m "feat: update _load_archive to return l3_bundles (backward-compat with old list format)"
```

---

## Task 2: Add `l3_agents` param, `_rank_by_l3`, `_inject_l3`, update `begin_context` and `end_context`

**Files:**
- Modify: `hpm/store/contextual_store.py`
- Test: `tests/store/test_contextual_store.py`

Background on key APIs:
- `extract_bundle(agent)` from `hpm.agents.hierarchical` returns `LevelBundle(agent_id, mu, weight, epistemic_loss)` — cache the result when reading multiple fields
- `encode_bundle(bundle)` returns `np.concatenate([bundle.mu, [bundle.weight, bundle.epistemic_loss]])` — shape `(feature_dim + 2,)`
- `make_pattern(mu, scale, pattern_type)` from `hpm.patterns.factory` constructs a GaussianPattern
- `_rank_by_l3` returns raw candidate objects (same objects from `_librarian.query_archive`) — access their archive path as `candidate.archive_path` (NOT `candidate.candidate.archive_path` — that wrapping only exists on forecaster-ranked results)
- The forecaster-ranked path uses `best.candidate.archive_path` because `ArchiveForecaster.rank()` wraps candidates in a `RankedCandidate` object

- [ ] **Step 1: Write the failing tests**

Add to `tests/store/test_contextual_store.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: l3_agents integration
# ---------------------------------------------------------------------------

def _make_mock_l3_agent(dim=10):
    """Minimal mock agent with an InMemoryStore for injection tests."""
    from hpm.store.memory import InMemoryStore
    from unittest.mock import MagicMock
    agent = MagicMock()
    agent.agent_id = "l3_mock"
    agent.store = InMemoryStore()
    agent.config = MagicMock()
    agent.config.feature_dim = dim
    return agent


def test_inject_l3_seeds_agent_store(tmp_path):
    """After begin_context with a matching L3 archive, L3 agent store contains seeded pattern."""
    import pickle, json, datetime
    from hpm.store.contextual_store import SubstrateSignature

    dim = 10
    tiered = TieredStore()
    mock_l3 = _make_mock_l3_agent(dim=dim)
    store = ContextualPatternStore(
        tiered_store=tiered,
        archive_dir=str(tmp_path),
        l3_agents=[mock_l3],
    )

    # Write an archive with an L3 bundle
    run_dir = tmp_path / store._run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    mu = np.linspace(0, 1, dim)
    tier2_p = make_pattern(mu=np.ones(8), scale=np.eye(8), pattern_type="gaussian")
    tiered._tier2.save(tier2_p, 1.0, "some_agent")
    tier2_state = tiered.query_tier2_all()
    archive_path = str(run_dir / "test_ctx.pkl")
    payload = {"tier2": tier2_state, "l3_bundles": [(mu, 0.8, 0.2)]}
    with open(archive_path, "wb") as f:
        pickle.dump(payload, f)

    # Write index so librarian can find this archive
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                             object_count=3, aspect_ratio_bucket="square")
    index_path = str(run_dir / "index.json")
    index = [{
        "context_id": "test_ctx",
        "signature": {
            "grid_size": [5, 5],
            "unique_color_count": 2,
            "object_count": 3,
            "aspect_ratio_bucket": "square",
        },
        "success_metrics": {"correct": True},
        "archive_path": archive_path,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }]
    with open(index_path, "w") as f:
        json.dump(index, f)

    # begin_context: should inject the L3 bundle into mock_l3.store
    store.begin_context(sig, first_obs=[np.zeros(8)])

    patterns = mock_l3.store.query_all()
    assert len(patterns) >= 1
    np.testing.assert_array_almost_equal(patterns[0][0].mu, mu)


def test_no_l3_agents_fallback_unchanged(tmp_path):
    """With l3_agents=None, begin_context behaviour is identical to existing code."""
    tiered = TieredStore()
    store = ContextualPatternStore(
        tiered_store=tiered,
        archive_dir=str(tmp_path),
        l3_agents=None,
    )
    from hpm.store.contextual_store import SubstrateSignature
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                             object_count=1, aspect_ratio_bucket="square")
    ctx_id = store.begin_context(sig, first_obs=[np.zeros(8)])
    assert isinstance(ctx_id, str)
```

- [ ] **Step 2: Run the tests — verify they fail**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_inject_l3_seeds_agent_store tests/store/test_contextual_store.py::test_no_l3_agents_fallback_unchanged -v
```

Expected: FAIL — `ContextualPatternStore.__init__` does not accept `l3_agents`.

- [ ] **Step 3: Add `l3_agents` to `__init__`**

In `hpm/store/contextual_store.py`, update the `__init__` signature and body:

```python
def __init__(
    self,
    tiered_store,
    archive_dir: str,
    run_id: Optional[str] = None,
    fingerprint_nll_threshold: float = 50.0,
    global_weight_threshold: float = 0.6,
    global_promotion_n: int = 5,
    l3_agents: list | None = None,
):
    self._store = tiered_store
    self._archive_dir = archive_dir
    self._run_id = run_id or str(uuid.uuid4())
    self._fingerprint_nll_threshold = fingerprint_nll_threshold
    self._global_weight_threshold = global_weight_threshold
    self._global_promotion_n = global_promotion_n
    self._last_sig: Optional[SubstrateSignature] = None
    self._l3_agents: list | None = l3_agents

    from hpm.store.archive_librarian import ArchiveLibrarian
    from hpm.store.archive_forecaster import ArchiveForecaster
    self._librarian = ArchiveLibrarian()
    self._forecaster = ArchiveForecaster()

    self._init_db()
```

- [ ] **Step 4: Add `_inject_l3` helper**

Add after `_load_index` (around line 221):

```python
def _inject_l3(self, l3_bundles: list) -> None:
    """Inject stored L3 bundle arrays as GaussianPattern seeds into L3 agents' stores.

    Each bundle is (mu_array, weight, epistemic_loss). Uses identity covariance —
    weight already encodes certainty; broad prior avoids over-constraining agents
    before they see the new task's training data.
    """
    if not self._l3_agents or not l3_bundles:
        return
    from hpm.patterns.factory import make_pattern
    for mu, weight, epistemic_loss in l3_bundles:
        pattern = make_pattern(mu=mu, scale=np.eye(len(mu)), pattern_type="gaussian")
        for agent in self._l3_agents:
            agent.store.save(pattern, weight, agent.agent_id)
```

- [ ] **Step 5: Add `_rank_by_l3` helper**

Add after `_inject_l3`:

```python
def _rank_by_l3(self, candidates: list, current_l3_vecs: list) -> list:
    """Re-rank candidates by cosine similarity of stored L3 bundles to current L3 state.

    candidates: raw candidate objects from ArchiveLibrarian (have .archive_path attribute)
    current_l3_vecs: list of encoded L3 bundle vectors (one per L3 agent)

    Candidates without L3 bundles (old-format archives) are moved to the end.
    Returns re-ranked candidates list (most similar first).

    Note: loads the full archive pkl per candidate (pickle does not support partial
    key loading). Acceptable at benchmark scale (hundreds of tasks).
    """
    def _score(candidate) -> float:
        try:
            with open(str(candidate.archive_path), "rb") as f:
                records = pickle.load(f)
        except Exception:
            return -1.0
        if isinstance(records, list):
            return -1.0  # old format — no L3 bundles
        l3_bundles = records.get("l3_bundles", [])
        if not l3_bundles or not current_l3_vecs:
            return -1.0
        sims = []
        for mu, _w, _eps in l3_bundles:
            norm_stored = np.linalg.norm(mu)
            for vec in current_l3_vecs:
                norm_cur = np.linalg.norm(vec)
                if norm_stored < 1e-12 or norm_cur < 1e-12:
                    continue
                sims.append(float(np.dot(mu, vec) / (norm_stored * norm_cur)))
        return float(np.mean(sims)) if sims else -1.0

    scored = [(candidate, _score(candidate)) for candidate in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]
```

- [ ] **Step 6: Update `begin_context` to use L3 ranking and injection**

Replace the existing `begin_context` method:

```python
def begin_context(self, sig: SubstrateSignature, first_obs: list) -> str:
    """Warm-start Tier 2 and L3 agents from best matching past episode; return context_id."""
    context_id = str(uuid.uuid4())
    self._last_sig = sig

    # Coarse filter: substrate signature
    candidates = self._librarian.query_archive(sig, Path(self._archive_dir))

    # Fine ranking: L3 cosine similarity if l3_agents available, else NLL (existing path).
    # Note: _rank_by_l3 returns raw candidate objects (candidate.archive_path).
    # The forecaster.rank() path returns RankedCandidate wrappers (best.candidate.archive_path).
    l3_bundles = []
    if self._l3_agents:
        from hpm.agents.hierarchical import extract_bundle, encode_bundle
        current_l3_vecs = [encode_bundle(extract_bundle(a)) for a in self._l3_agents]
        ranked_candidates = self._rank_by_l3(candidates, current_l3_vecs)
        if ranked_candidates:
            best = ranked_candidates[0]
            l3_bundles = self._load_archive(str(best.archive_path))
    else:
        ranked = self._forecaster.rank(
            candidates, first_obs, nll_threshold=self._fingerprint_nll_threshold
        )
        if ranked:
            best = ranked[0]
            l3_bundles = self._load_archive(str(best.candidate.archive_path))

    # Inject L3 bundles as seed patterns (no-op if l3_agents=None or l3_bundles=[])
    self._inject_l3(l3_bundles)

    # Global patterns injection (unchanged)
    self._inject_globals()

    # Open Tier 1 context
    self._store.begin_context(context_id)
    return context_id
```

- [ ] **Step 7: Update `end_context` to write dict-format archive when l3_agents present**

Replace lines 162-169 in `end_context` (the archive write block):

```python
archive_path = os.path.join(run_dir, f"{context_id}.pkl")
tmp_path_pkl = archive_path + ".tmp.pkl"
tier2_state = self._store.query_tier2_all()

correct = success_metrics.get("correct", False)
if self._l3_agents and correct:
    from hpm.agents.hierarchical import extract_bundle
    l3_bundles = []
    for a in self._l3_agents:
        b = extract_bundle(a)  # cache result — read all fields from same bundle
        l3_bundles.append((b.mu.copy(), float(b.weight), float(b.epistemic_loss)))
    payload = {"tier2": tier2_state, "l3_bundles": l3_bundles}
else:
    payload = tier2_state  # old list format when no l3_agents or task failed

with open(tmp_path_pkl, "wb") as f:
    pickle.dump(payload, f)
os.replace(tmp_path_pkl, archive_path)
```

Note: `tmp_path_pkl` (not `tmp_path`) avoids shadowing the outer `tmp_path` variable name if present.

- [ ] **Step 8: Run the new tests**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_inject_l3_seeds_agent_store tests/store/test_contextual_store.py::test_no_l3_agents_fallback_unchanged -v
```

Expected: PASS

- [ ] **Step 9: Run the full test suite**

```bash
python3 -m pytest tests/ -q --tb=short
```

Expected: 651 passed, 9 skipped (no regressions).

- [ ] **Step 10: Commit**

```bash
git add hpm/store/contextual_store.py tests/store/test_contextual_store.py
git commit -m "feat: add l3_agents support to ContextualPatternStore for conceptual warm-starting"
```

---

## Task 3: Restructure `run_persistent()` and wire `l3_agents`

`l1_contextual` must be built AFTER `stacked_orch` to access `stacked_orch.level_agents[-1]`. Currently `l1_contextual` is built before `stacked_orch` in `run_persistent()`. This task restructures the construction order.

**Files:**
- Modify: `benchmarks/hierarchical_arc.py`

Key insight: `ContextualPatternStore` wraps `TieredStore` and delegates all `save()`/`query()` calls to it. Agents don't need to be constructed with `l1_contextual` as their store — they can use `l1_tiered` directly, and the benchmark harness calls `l1_contextual.begin_context()`/`end_context()` for lifecycle management. This matches the `multi_agent_arc.py` pattern exactly.

New construction order in `run_persistent()`:
1. Create `l1_tiered`, `flat_tiered`
2. Build `l1_orch` with `store=l1_tiered` (not l1_contextual)
3. Build `flat_orch` with `store=flat_tiered`
4. Build `stacked_orch` (uses l1_orch)
5. Build `l1_contextual` with `l3_agents=stacked_orch.level_agents[-1]` (wraps l1_tiered)
6. Build `flat_contextual` (wraps flat_tiered, no l3_agents)

- [ ] **Step 1: Read the current construction order**

```bash
grep -n "stacked_orch\|l1_contextual\|flat_contextual\|l1_tiered\|flat_tiered\|l1_orch\|flat_orch\|ContextualPatternStore" benchmarks/hierarchical_arc.py
```

Confirm current order: `l1_tiered` → `flat_tiered` → `l1_contextual` → `flat_contextual` → `l1_orch` → `flat_orch` → `stacked_orch`. Restructure so `l1_contextual` and `flat_contextual` are created AFTER `stacked_orch`.

- [ ] **Step 2: Restructure `run_persistent()`**

Replace the construction block in `run_persistent()`. The new order:

```python
import os
from hpm.store.tiered_store import TieredStore
from hpm.store.contextual_store import ContextualPatternStore, extract_signature
from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator
from hpm.agents.stacked import StackedOrchestrator

tasks = load_tasks()
tasks = [t for t in tasks if task_fits(t)][:400]
rng = np.random.default_rng(42)

os.makedirs("data", exist_ok=True)
l1_tiered = TieredStore()
flat_tiered = TieredStore()

# Build L1 orchestrators using bare TieredStores.
# ContextualPatternStore wraps l1_tiered for lifecycle management (begin/end_context).
# Agents use l1_tiered directly — ContextualPatternStore delegates all save/query calls to it.
l1_orch, l1_agents, _ = _make_persistent_l1_orchestrator(store=l1_tiered)
flat_orch, flat_agents, _ = _make_persistent_flat_orchestrator(store=flat_tiered)

# Build L2/L3 and assemble stacked orchestrator
_upper_configs = STACK_CONFIGS[1:]
_upper_orch, _upper_agents = make_stacked_orchestrator(
    l1_feature_dim=L1_FEATURE_DIM + 2,
    level_configs=_upper_configs,
)
level_orches = [l1_orch] + _upper_orch.level_orches
level_agents = [l1_agents] + _upper_orch.level_agents
level_Ks = [1] + [cfg.K for cfg in STACK_CONFIGS[1:]]
stacked_orch = StackedOrchestrator(level_orches, level_agents, level_Ks)

# Build ContextualPatternStores AFTER stacked_orch so we can pass l3_agents
l1_contextual = ContextualPatternStore(
    tiered_store=l1_tiered,
    archive_dir="data/archives/hierarchical_arc_persistent",
    l3_agents=stacked_orch.level_agents[-1],   # L3 agents for conceptual retrieval
)
flat_contextual = ContextualPatternStore(
    tiered_store=flat_tiered,
    archive_dir="data/archives/hierarchical_arc_persistent_flat",
)

recombinator = CrossTaskRecombinator()
```

- [ ] **Step 3: Smoke test — run 5 tasks**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import benchmarks.hierarchical_arc as ha
import benchmarks.multi_agent_arc as ma
orig_load = ma.load_tasks
ma.load_tasks = lambda: orig_load()[:5]
ha.TRAIN_REPS = 2
m = ha.run_persistent()
print('OK:', {k: v for k, v in m.items() if k != 'l3_ticks'})
" 2>&1 | grep -v "Warning\|unauthenticated\|HF_TOKEN"
```

Expected: prints dict with `n_tasks=5`. No exception.

- [ ] **Step 4: Run the full test suite**

```bash
python3 -m pytest tests/ -q --tb=short
```

Expected: 651 passed, 9 skipped.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/hierarchical_arc.py
git commit -m "feat: restructure run_persistent() to wire l3_agents into ContextualPatternStore"
```

- [ ] **Step 6: Run the full benchmark and verify L3 archive grows**

```bash
rm -rf data/archives/hierarchical_arc_persistent data/archives/hierarchical_arc_persistent_flat
python3 benchmarks/hierarchical_arc.py --persistent
```

Then verify L3 archive has entries:

```bash
ls data/archives/hierarchical_arc_persistent/*/*.pkl 2>/dev/null | wc -l
```

Expected: ≥ 10 pkl files (at least 10 tasks solved correctly with L3 bundles saved).

- [ ] **Step 7: Push**

```bash
git push
```
