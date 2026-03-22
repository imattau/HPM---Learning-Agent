# Contextual Pattern Store

**Date:** 2026-03-22

## Goal

Solve catastrophic forgetting and the absence of warm-start recall in the non-stationary ARC benchmark environment. Each ARC episode presents a structurally distinct task; the current `TieredStore` starts cold every episode. `ContextualPatternStore` wraps `TieredStore` to archive and retrieve Tier 2 state across episodes, enabling warm-start from structurally similar past tasks and progressive accumulation of globally useful patterns.

## Scope

- **In scope:** `ContextualPatternStore` wrapper, `SubstrateSignature` extraction, pickle-based archive, SQLite global pattern promotion, `Librarian` and `Forecaster` specialist classes, ARC benchmark integration, unit and integration tests
- **Out of scope:** Changes to `TieredStore` internals or `Agent.step()` dynamics, cross-run archive sharing, Tier 1 archiving, distributed or async archive writes

---

## Section 1: Phase 1 — Archive and Warm-Start

### `SubstrateSignature` (`hpm/store/contextual_store.py`)

A frozen dataclass capturing the structural fingerprint of an ARC task grid:

```python
@dataclass(frozen=True)
class SubstrateSignature:
    grid_size: tuple[int, int]
    unique_color_count: int
    object_count: int
    aspect_ratio_bucket: str  # "square" | "landscape" | "portrait"
```

`extract_signature(grid: np.ndarray) -> SubstrateSignature` is a standalone module-level function. It counts unique non-background colours, counts contiguous objects (4-connected components), and assigns `aspect_ratio_bucket` based on `rows / cols` (square: 0.8–1.25, landscape: < 0.8, portrait: > 1.25).

### `ContextualPatternStore` class (`hpm/store/contextual_store.py`)

Wraps a `TieredStore` instance. All `TieredStore` public methods are delegated unchanged — the wrapper is transparent to `Agent.step()`. The wrapper adds two lifecycle methods called explicitly by the benchmark harness:

**`begin_context(sig: SubstrateSignature, first_obs: list[np.ndarray]) -> str`**

`first_obs` is a list of up to 3 observation arrays from the start of the current episode. The benchmark harness collects these before calling `begin_context`.

1. Coarse filter: query `index.json` for archived episodes whose `grid_size` matches exactly and whose `unique_color_count` is within ±1.
2. Fine filter: for each coarse-filter candidate, deserialise the archive and compute the mean Pattern Fingerprint NLL of the archive's Tier 2 patterns against `first_obs`. Retain candidates whose mean NLL is below `fingerprint_nll_threshold` (default `50.0`; lower NLL = better match). Select the lowest-NLL survivor.
3. If a match is found: deserialise that archive's Tier 2 state into the wrapped `TieredStore`. If no match: leave `TieredStore` in its default fresh state.
4. Unconditionally inject all `is_global=True` patterns from `SQLiteStore` into Tier 2 (Phase 2 behaviour; in Phase 1 this is a no-op because no global patterns exist yet).
5. Return a `context_id` (UUID string) for this episode.

**`end_context(context_id: str, success_metrics: dict) -> None`**

1. Serialise the wrapped `TieredStore`'s Tier 2 state to `data/archives/<run_id>/<context_id>.pkl`.
2. Append an entry to `data/archives/<run_id>/index.json` recording `context_id`, `signature` fields, `success_metrics`, and timestamp.
3. Run the Global Pass (Phase 2; no-op in Phase 1).

### Archive layout

```
data/archives/
  <run_id>/
    index.json          # list of {context_id, signature, success_metrics, timestamp}
    <context_id>.pkl    # pickle of TieredStore Tier 2 state dict
```

`run_id` is assigned at `ContextualPatternStore.__init__` time (UUID or caller-supplied string). Archive writes are atomic: write to `<context_id>.tmp.pkl`, then `os.replace` to the final path.

### Benchmark integration

In the ARC benchmark (`benchmarks/multi_agent_arc.py`), `--persistent` mode currently constructs a bare `TieredStore`. This is replaced with `ContextualPatternStore(archive_dir="data/archives", ...)`. The harness:

1. Collects the first 3 observations from the episode's input grid before calling `begin_context(sig, first_obs)`.
2. Calls `end_context(context_id, metrics)` after the episode completes.

Non-persistent mode is unaffected.

### Tests (`tests/store/test_contextual_store.py`)

- `extract_signature` returns correct `grid_size`, `unique_color_count`, `object_count`, and `aspect_ratio_bucket` for known inputs
- Coarse filter excludes archives with mismatched `grid_size` or `color_count` outside ±1
- Fine filter selects the lowest-NLL candidate and rejects candidates whose NLL exceeds `fingerprint_nll_threshold`
- Round-trip: `end_context` writes a file; `begin_context` on a matching signature deserialises Tier 2 and confirms patterns are present
- Integration: agent runs task A, `end_context` archives it, agent runs task B with matching signature, `begin_context` warm-starts from task A's Tier 2 state

---

## Section 2: Phase 2 — Global Pass and `is_global`

### SQLiteStore schema additions

Two columns are added to the existing patterns table via idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migration run at `ContextualPatternStore.__init__`:

```sql
ALTER TABLE patterns ADD COLUMN IF NOT EXISTS is_global BOOLEAN DEFAULT 0;
ALTER TABLE patterns ADD COLUMN IF NOT EXISTS context_ids TEXT DEFAULT '[]';
```

`context_ids` stores a JSON-encoded list of episode context IDs in which this pattern's weight exceeded `global_weight_threshold`.

### Global Pass (in `end_context`)

After archiving Tier 2:

1. For each pattern in Tier 2 with `weight > global_weight_threshold` (default `0.6`): upsert into `SQLiteStore` matched on pattern `id`. On insert (new pattern): write all fields. On update (pattern already exists): update `mu` and `weight` in place; append the current `context_id` to `context_ids`. The `id` field is the primary key; no other field is used for matching.
2. If `len(context_ids) >= global_promotion_n` (default `5`): set `is_global = True` for that pattern.

### Warm-start update (in `begin_context`)

After the archive warm-start step, unconditionally load all `SQLiteStore` patterns where `is_global = True` into Tier 2. These patterns are injected regardless of whether an archive match was found.

### New `AgentConfig` fields (`hpm/config.py`)

```python
global_weight_threshold: float = 0.6
global_promotion_n: int = 5
```

Defaults leave all existing behaviour unchanged.

### Benchmark signal

After Phase 2, the benchmark should report `global_patterns_loaded > 0` after episode 5 (assuming at least one pattern has appeared in 5 episodes). Performance on episode 6+ should be stable or improving compared to a cold-start baseline.

---

## Section 3: Phase 3 — Specialist Roles (Pure Refactor)

Phase 3 is a structural refactor with no change to observable behaviour or benchmark metrics.

### `CandidateArchive` dataclass

```python
@dataclass
class CandidateArchive:
    context_id: str
    signature: SubstrateSignature
    archive_path: str   # absolute path to the .pkl file
    success_metrics: dict
```

Used as the exchange type between `Librarian` and `Forecaster`.

### `Librarian` (`hpm/agents/librarian.py`)

Stateless class (no `__init__` state beyond config). Owns:
- `extract_signature(grid) -> SubstrateSignature`
- `query_archive(sig, archive_dir) -> list[CandidateArchive]` — applies the coarse filter and returns matching candidates
- `run_global_pass(tier2_patterns, context_id, sqlite_store, config) -> None`
- `write_global_flags(sqlite_store, config) -> None`

### `Forecaster` (`hpm/agents/forecaster.py`)

Stateless class. Owns:
- `rank(candidates: list[CandidateArchive], obs: list[np.ndarray]) -> list[RankedCandidate]`

`RankedCandidate` is a dataclass: `{candidate: CandidateArchive, nll: float}`, sorted ascending by NLL. Candidates whose NLL exceeds `fingerprint_nll_threshold` are excluded from the result.

### Agent refactor

`Agent` becomes a pure Actor. Its `step()` loop is unchanged. `begin_context()` becomes an explicit call chain:

```python
candidates = librarian.query_archive(sig, archive_dir)
ranked = forecaster.rank(candidates, first_obs)
self._load_archive(ranked[0] if ranked else None)
```

The `ContextualPatternStore` delegates to `Librarian` and `Forecaster` rather than containing that logic itself.

### Constraint

No benchmark metric changes between Phase 2 and Phase 3. The refactor is validated by confirming all existing Phase 2 tests pass without modification.

---

## Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| Wrapper (Option C) not refactor | Zero risk to battle-tested TieredStore dynamics; wrapper is transparent to Agent.step() |
| Pickle Tier 2 only (not Tier 1) | Tier 1 is ephemeral working memory; Tier 2 holds promoted, stable patterns worth recalling |
| SQLiteStore for globals | Unified schema, single source of truth, fits HPM's promotion-ladder semantics |
| Coarse filter then fine NLL filter | Coarse filter is O(1) per candidate (index scan); NLL test on first 3 obs adds precision without circular dependency on episode outcome |
| `first_obs` passed into `begin_context` | Avoids circular dependency; benchmark harness has the observations before `begin_context` is called |
| `fingerprint_nll_threshold = 50.0` | Tunable default; chosen as a conservative starting point above typical within-task NLL variance |
| Upsert by pattern `id`, update mu+weight | Single source of truth per pattern; `id` is stable across episodes for the same learned structure |
| Atomic archive writes (tmp + replace) | Prevents corrupt index from a mid-write crash |
| Phase-by-phase validation | Each phase benchmarked before the next; applies HPM's own gating logic to the development process |
| Phase 3 is pure refactor | Separating Librarian and Forecaster improves testability and boundary clarity without touching any dynamics |
