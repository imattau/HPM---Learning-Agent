# TieredForest — Design Spec

**Date:** 2026-03-29
**Status:** Approved

## Overview

`TieredForest` is a drop-in subclass of `Forest` that adds two-level memory tiering: a RAM-resident hot cache (LRU-capped, psutil-monitored) and a disk-based cold store (`.npz` files). It solves the OOM failure observed in the NLP experiment where unbounded node accumulation exhausted RAM, while preserving Observer semantics for the active (hot) node population.

---

## 1. Motivation

The Observer accumulates learned nodes indefinitely. Each node at D=428 stores a 428×428 float64 sigma matrix (~1.47MB). After two passes over 2000 NLP observations, 864+ learned nodes were created (~1.27GB for learned sigmas alone). Three passes OOM-killed the process at ~5.5GB.

The fix must not change what the Observer can learn from hot nodes — only where dormant nodes live.

---

## 2. Architecture

```
TieredForest
├── _mu_index: dict[str, np.ndarray]   # ALL node mus — always in RAM
├── _hot: OrderedDict[str, HFN]        # LRU cache of full nodes — capped
├── _cold_dir: Path                    # disk store: {node_id}.npz per node
└── _protected_ids: set[str]           # never deleted (priors)
```

**Key principle:** `_mu_index` holds mu vectors for every node (hot and cold). At ~3.4KB per node, 10,000 nodes ≈ 34MB — always affordable. Full nodes (mu + sigma) live in `_hot` or on disk. Distance screening always uses `_mu_index`; sigma is only loaded when a node is a top-k candidate.

---

## 3. Active vs Dormant Nodes — HPM Rationale

`active_nodes()` returns **hot nodes only**. This is an intentional design choice, not a limitation:

- The Observer's weight updates, score updates, absorption checks, and density calculations all operate on `active_nodes()`. Cold nodes are excluded from these passes.
- This is HPM-aligned: **dormant patterns do not consume active processing resources**. A pattern that hasn't been relevant to recent observations is not updated, not absorbed, and does not contribute to density — analogous to a rarely-used memory trace that fades from working memory.
- Cold nodes re-enter the active set when `retrieve()` finds them as top-k candidates for a new observation, at which point they are promoted to hot and resume participation in all Observer operations.
- This means no node is permanently frozen — only nodes that remain irrelevant for extended periods stay cold.

**Consequence:** a cold node cannot absorb a hot node (or vice versa) during a pass where it is cold. This is acceptable: if a node is cold, it is not currently explaining observations, so absorption decisions involving it are not urgent.

---

## 4. Node Lifecycle

```
Created → _hot + _mu_index
  │
  ├─ len(_hot) > max_hot → evict LRU to _cold_dir (both protected and unprotected)
  │
  ├─ sweep (every sweep_every obs):
  │    ├─ psutil check: free RAM < min_free_ram_mb?
  │    │    └─ yes → sort hot by persistence (asc), evict bottom half to cold
  │    │         (protected: cold only; unprotected: cold only during emergency)
  │    │
  │    └─ persistence floor scan (operates on post-emergency hot set):
  │         ├─ protected + below floor → evict to cold
  │         └─ unprotected + below floor → delete entirely (hot, mu_index, .npz)
  │
  └─ retrieve() finds cold node in top-k → promote to hot (evict LRU hot if needed)
```

**Deregister policy:**
- Protected node → no-op regardless of tier: no changes to `_hot`, `_mu_index`, or `.npz`
- Unprotected node → remove from `_hot`, `_mu_index`, delete `.npz` if present (whether hot or cold)

> **Protected nodes and eviction:** The sweep may evict a protected node from `_hot` to cold (LRU pressure or emergency RAM). This is system-initiated and permitted — it only moves the node, never deletes it. Caller-initiated `deregister()` on a protected node is always a complete no-op. The distinction is: sweep = may move to cold; deregister = no effect at all.

**On `__init__`:** delete all `.npz` files in `cold_dir` from any previous run (no crash recovery — fresh start each time).

---

## 5. Interface

```python
class TieredForest(Forest):
    def __init__(
        self,
        D: int,
        forest_id: str,
        cold_dir: str | Path,
        max_hot: int = 500,
        persistence_floor: float = 0.1,
        sweep_every: int = 100,
        min_free_ram_mb: int = 500,
        protected_ids: set[str] | None = None,
    ): ...

    # Forest interface overrides
    def register(self, node: HFN) -> None: ...
    def deregister(self, node_id: str) -> None: ...
    def active_nodes(self) -> list[HFN]: ...        # hot nodes only (see Section 3)
    def retrieve(self, x, k: int) -> list[HFN]: ... # mu_index screen → load top-k
    def __contains__(self, node_id: str) -> bool: ...  # checks _mu_index (hot + cold)
    def __len__(self) -> int: ...                    # total count: hot + cold

    # New methods
    def get(self, node_id: str) -> HFN: ...         # load from cold if needed
    def set_protected(self, ids: set[str]) -> None: ...
    def hot_count(self) -> int: ...
    def cold_count(self) -> int: ...
```

**`__contains__`** checks `_mu_index` — returns True for both hot and cold nodes. This ensures the Observer's `ids[0] not in self.forest` membership tests (observer.py lines 495, 498, 499) correctly detect all known nodes.

**`__len__`** returns `len(_mu_index)` — total node count (hot + cold). This ensures `f"leaf_{len(self.forest)}"` ID generation in observer.py (line 482) counts all nodes, not just hot ones.

> **Note:** `leaf_{len(forest)}` as a node ID is a pre-existing issue in observer.py that can produce collisions after deletions. Not introduced by this spec; not fixed here.

**`get(node_id)`** returns a full HFN, loading from cold if not in `_hot`. Used to replace the two direct `forest._registry[id]` accesses in observer.py.

---

## 6. `_sync_gaussian()` Override

`Forest._sync_gaussian()` stacks all sigma matrices to compute the Forest's own Gaussian summary (`forest.mu`, `forest.sigma`). Loading all cold sigmas would defeat the purpose of tiering.

`TieredForest` overrides `_sync_gaussian()` to compute from **hot nodes only**. This is an approximation — the Forest's summary Gaussian reflects the active population, not the full forest. The Observer does not use `forest.mu` or `forest.sigma` directly (confirmed by inspection of observer.py), so this approximation has no effect on learning dynamics.

```python
def _sync_gaussian(self) -> None:
    """Approximate: sync from hot nodes only. Cold nodes excluded."""
    hot = list(self._hot.values())
    if not hot:
        self.mu = np.zeros(self._D)
        self.sigma = np.eye(self._D)
    else:
        mus = np.stack([n.mu for n in hot])
        self.mu = mus.mean(axis=0)
        sigmas = np.stack([n.sigma for n in hot])
        self.sigma = sigmas.mean(axis=0)
```

---

## 7. Sweep Behaviour

Called every `sweep_every` observations (default 100). Steps execute in order on the same sweep pass:

1. **Emergency RAM check:** call `psutil.virtual_memory().available`. If below `min_free_ram_mb`:
   - Sort all hot nodes by persistence score (ascending). Protected and unprotected nodes are sorted together.
   - Evict the bottom half to cold (write `.npz`, remove from `_hot`). Both protected and unprotected nodes may be evicted to cold here. Neither is deleted from `_mu_index`.

2. **Persistence floor scan** (operates on the post-emergency hot set):
   - For each hot node below `persistence_floor`:
     - If protected: evict to cold
     - If unprotected: delete entirely (`_hot`, `_mu_index`, `.npz`)

---

## 8. Serialisation

Each node saved as `{cold_dir}/{node_id}.npz` using `np.savez_compressed` (numpy's native binary format — no pickle). Fields stored: `mu`, `sigma`, `id` (as a length-1 string array). These are the complete HFN data fields — `persistence` is computed on-the-fly by `hfn.fractal.persistence_scores` and is not stored on the node. `_children` and `_edges` are not serialised; cold nodes are treated as leaves and re-acquire structural relationships when promoted back to hot. Loaded via `np.load()`. Approximately 1–2ms per node on SSD; only occurs during eviction and promotion.

---

## 9. Changes to Existing Code

**`hfn/observer.py`** — two lines only:
```python
# Before (lines 510, 521):
self.forest._registry[ids[0]].mu
self.forest._registry[ids[0]]
# After:
self.forest.get(ids[0]).mu
self.forest.get(ids[0])
```

**World model builders** (`build_nlp_world_model`, `build_dsprites_world_model`) — accept optional `forest_cls` parameter defaulting to `Forest`. Extra kwargs are forwarded to `forest_cls` constructor only when `forest_cls is not Forest`; ignored otherwise:
```python
def build_nlp_world_model(
    forest_cls=Forest,
    **tiered_kwargs,
) -> tuple[Forest, set[str]]:
    kwargs = tiered_kwargs if forest_cls is not Forest else {}
    forest = forest_cls(D=D, forest_id="nlp", **kwargs)
    ...
```

**Experiment scripts** — swap to TieredForest:
```python
from hfn.tiered_forest import TieredForest
forest, prior_ids = build_nlp_world_model(
    forest_cls=TieredForest,
    cold_dir="/tmp/hfn_nlp_cold",
    max_hot=500,
)
```

---

## 10. File Structure

```
hfn/
  tiered_forest.py        — new: TieredForest class
  observer.py             — 2-line fix (_registry[id] → get(id))
  forest.py               — unchanged
tests/
  hfn/
    test_tiered_forest.py — new: unit tests
```

---

## 11. Sizing Reference

| D | Per-node sigma (float64) | max_hot=500 RAM | 10k nodes mu_index |
|---|---|---|---|
| 256 (dSprites) | 0.52MB | 260MB | 12MB |
| 428 (NLP) | 1.47MB | 735MB | 34MB |

`min_free_ram_mb=500` ensures the OS and other processes always have headroom. Emergency eviction triggers before the kernel OOM killer does.

---

## 12. Dependencies

- `psutil` — system RAM monitoring (new dependency, `pip install psutil`)
- `numpy` — already present (serialisation via `np.savez_compressed`)
