# TieredForest — Design Spec

**Date:** 2026-03-29
**Status:** Approved

## Overview

`TieredForest` is a drop-in subclass of `Forest` that adds two-level memory tiering: a RAM-resident hot cache (LRU-capped, psutil-monitored) and a disk-based cold store (`.npz` files). It solves the OOM failure observed in the NLP experiment where unbounded node accumulation exhausted RAM, while preserving all Observer semantics.

---

## 1. Motivation

The Observer accumulates learned nodes indefinitely. Each node at D=428 stores a 428×428 float64 sigma matrix (~1.47MB). After two passes over 2000 NLP observations, 864+ learned nodes were created (~1.27GB for learned sigmas alone). Three passes OOM-killed the process at ~5.5GB.

The fix must not change what the Observer can learn — only where nodes live.

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

## 3. Node Lifecycle

```
Created → _hot + _mu_index
  │
  ├─ len(_hot) > max_hot → evict LRU to _cold_dir
  │
  ├─ sweep: persistence < floor AND not protected → delete entirely
  ├─ sweep: persistence < floor AND protected → evict to cold only
  ├─ sweep: psutil free RAM < min_free_ram_mb → emergency evict bottom half by persistence
  │
  └─ retrieve() finds cold node in top-k → load from cold → promote to _hot
```

**Deregister policy:**
- Protected node → no-op (silently ignored at storage layer)
- Unprotected node → remove from `_hot`, `_mu_index`, delete `.npz` if present

---

## 4. Interface

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

    def register(self, node: HFN) -> None: ...
    def deregister(self, node_id: str) -> None: ...
    def active_nodes(self) -> list[HFN]: ...       # hot nodes only
    def retrieve(self, x, k: int) -> list[HFN]: ... # mu_index screen → load top-k
    def get(self, node_id: str) -> HFN: ...         # load from cold if needed
    def set_protected(self, ids: set[str]) -> None: ...
    def hot_count(self) -> int: ...
    def cold_count(self) -> int: ...
```

**`active_nodes()`** returns hot nodes only. The Observer's absorption, recombination, and fractal passes operate on the hot set. Nodes cycle through hot as observations arrive — no node is permanently frozen.

**`retrieve(x, k)`** screens `_mu_index` by Euclidean distance, loads any cold top-k candidates into `_hot` (evicting LRU if needed), and returns full HFN objects. This ensures the closest nodes are always considered regardless of hot/cold status.

**`get(node_id)`** returns a full HFN, loading from cold if not in `_hot`. Used by the two lines in `observer.py` that currently access `forest._registry[id]` directly.

---

## 5. Sweep Behaviour

Called every `sweep_every` observations (default 100):

1. Check `psutil.virtual_memory().available`. If below `min_free_ram_mb`:
   - Sort hot nodes by persistence score (ascending)
   - Evict bottom half to cold (skip protected nodes — they are evicted but not deleted)
2. For each hot node below `persistence_floor`:
   - If protected: evict to cold
   - If unprotected: delete entirely (`_hot`, `_mu_index`, `.npz`)

Sweep adds one `psutil` call per `sweep_every` observations — negligible overhead.

---

## 6. Serialisation

Each node saved as `{cold_dir}/{node_id}.npz` using `np.savez_compressed` (numpy's native binary format — no pickle). Fields stored: `mu`, `sigma`, `id`, `persistence`, `n_observations`. Loaded via `np.load()`. Approximately 1–2ms per node on SSD; only occurs during eviction and promotion.

---

## 7. Changes to Existing Code

**`hfn/observer.py`** — two lines only:
```python
# Before (lines 510, 521):
self.forest._registry[ids[0]].mu
self.forest._registry[ids[0]]
# After:
self.forest.get(ids[0]).mu
self.forest.get(ids[0])
```

**World model builders** (`build_nlp_world_model`, `build_dsprites_world_model`) — accept optional `forest_cls` parameter defaulting to `Forest`:
```python
def build_nlp_world_model(
    forest_cls=Forest,
    **tiered_kwargs,
) -> tuple[Forest, set[str]]: ...
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

## 8. File Structure

```
hfn/
  tiered_forest.py        — new: TieredForest class
  observer.py             — 2-line fix
  forest.py               — unchanged
tests/
  hfn/
    test_tiered_forest.py — new: unit tests
```

---

## 9. Sizing Reference

| D | Per-node sigma (float64) | max_hot=500 RAM | 10k nodes mu_index |
|---|---|---|---|
| 256 (dSprites) | 0.52MB | 260MB | 12MB |
| 428 (NLP) | 1.47MB | 735MB | 34MB |

`min_free_ram_mb=500` ensures the OS and other processes always have headroom. Emergency eviction triggers before the kernel OOM killer does.

---

## 10. Dependencies

- `psutil` — system RAM monitoring (new dependency, `pip install psutil`)
- `numpy` — already present (serialisation via `np.savez_compressed`)
