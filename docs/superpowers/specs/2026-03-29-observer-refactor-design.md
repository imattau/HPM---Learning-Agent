# Observer Refactor — Evaluator + Recombination Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## 1. Motivation

The Observer currently combines three distinct responsibilities:

1. **Expansion + weight/score dynamics** — perception and pattern reinforcement (HPM layer 2)
2. **Structural evaluation** — deciding which patterns are worth keeping (HPM layer 3)
3. **Structural execution** — absorbing and compressing nodes (mechanical operations)

The Observer docstring already notes this: "absorption/compression should move up to the HPM AI's evaluator layer." This refactor makes that split explicit by introducing two collaborators — **Evaluator** and **Recombination** — that Observer calls independently.

This is not purely a cleanup. Evaluator becomes the HPM layer 3 component in full: it will hold all evaluator/gatekeeper functions the HPM framework requires, including gap detection that will later feed into external substrate/field queries. The Observer becomes capable of knowing what it doesn't know.

---

## 2. Architecture

```
Observer
  ├── calls → Evaluator   (pure queries, returns metrics — no side effects)
  └── calls → Recombination   (structural executor — returns new HFN)
```

**Observer** remains the coordinator and tracker. It holds all mutable state (weights, scores, co-occurrence, miss counts), makes all decisions, and drives the full observe() loop. It calls Evaluator to get metrics and gap signals, uses those to decide what to do, then calls Recombination to execute structural changes.

**Evaluator** is a pure evaluation service — stateless, no Forest mutation, no decisions. Takes nodes and Observer state as inputs, returns numeric results. Owns all three evaluation classes (fractal geometry, gap detection, HPM framework evaluations).

**Recombination** is a structural executor. Given two nodes and a Forest, it performs the merge operation and returns the new HFN. No decision logic — Observer decides when; Recombination executes how.

HPM strategy flags (`persistence_guided_absorption`, `multifractal_guided_absorption`, `lacunarity_guided_creation`, `hausdorff_absorption_threshold`, etc.) stay on Observer — they are Observer configuration for when to invoke which Evaluator method.

---

## 3. Evaluator

`hfn/evaluator.py` — new file.

The Evaluator has three responsibility classes:

### 3.1 Fractal Geometry Evaluations

Pure queries against the current node set using the existing `hfn.fractal` tools.

```python
def crowding(node_mu, nodes, radius) -> int
```
Count non-protected nodes within `radius` of `node_mu`. Used by multifractal-guided absorption. (Moves from `Observer._local_crowding`.)

```python
def density_ratio(x, nodes, radius) -> float
```
Ratio of local node density at `x` to mean density. Used by lacunarity-guided creation suppression. (Moves from `Observer._local_density_ratio`.)

```python
def nearest_prior_dist(mu, prior_mus) -> float
```
Euclidean distance from `mu` to the nearest prior node. Used to rank compression candidates. (Moves from `Observer._nearest_prior_dist` + `_build_prior_mus`.)

```python
def hausdorff_candidates(nodes, weights, threshold, weight_floor, protected_ids) -> list[tuple[HFN, HFN]]
```
Returns `(node, best_match)` pairs where node is close to a better-weighted node and below the weight floor. Observer decides whether to absorb each pair.

```python
def persistence_scores(nodes, weights) -> dict[str, float]
```
Thin wrapper over `hfn.fractal.persistence_scores`. Returns persistence score per node id.

### 3.2 Gap / Unknown Detection

Queries that tell the Observer what its model does not yet cover. These use fractal geometry on the current Forest state — no separate history needed; the living node set implicitly encodes what has been learned.

```python
def coverage_gap(x, nodes, radius) -> float
```
How sparse is the region around observation `x` given current nodes. High value = Observer has little knowledge near this point. Used by Observer to raise readiness to create a new node.

```python
def underrepresented_regions(nodes) -> list[np.ndarray]
```
Regions of observation space with low node density, derived from lacunarity analysis across the full node set. Returns representative mu-vectors for each gap region. Observer can expose these to external substrates/fields as "I need more observations here."

### 3.3 HPM Framework Evaluations

The HPM evaluator/gatekeeper layer: determining which patterns are worth keeping, signalling novelty/redundancy, and providing hooks for external reinforcement. These are needed for dynamic learning and will be extended as the HPM AI grows.

```python
def accuracy(x, node) -> float
```
Functional utility of a node for a specific observation. (Moves from `Observer._accuracy`.)

```python
def description_length(node) -> float
```
Complexity proxy for a node. Thin wrapper over `node.description_length()`. Used in score computation.

```python
def score(x, node, lambda_complexity) -> float
```
Combined utility evaluation: `accuracy(x, node) - lambda_complexity * description_length(node)`. (Moves the score formula out of `Observer._update_scores`.)

```python
def coherence(node) -> float
```
Internal consistency of a node's pattern. Derived from sigma matrix properties (condition number or eigenvalue spread — low spread = high coherence). Returns value in [0, 1].

```python
def curiosity(x, nodes, weights) -> float
```
Novelty signal for observation `x`: how far is `x` from well-weighted nodes? High value = the Observer has not seen patterns like this. Complements `coverage_gap` (geometric) with a weight-aware novelty measure.

```python
def boredom(node, weights, scores) -> float
```
Redundancy signal for a node: high weight but low accuracy improvement over recent observations = this node is over-represented. Signals that the Observer is not learning from this node anymore.

```python
def reinforcement_signal(node_id) -> float
```
Hook for external reward/reinforcement input. Default implementation returns 0.0 (no external signal). Can be subclassed or injected to connect external evaluators (reward models, social context, field signals). Observer uses this in weight update decisions.

---

## 4. Recombination

`hfn/recombination.py` — new file.

Pure structural executor. No decision logic. Takes nodes and a Forest, performs the merge, returns the new node. Observer updates its own weights/scores for the new node after the call.

```python
def absorb(node: HFN, target: HFN, forest: Forest) -> HFN
```
Recombines `node` into `target`. Deregisters both from Forest, registers merged node, returns it. Observer transfers `target`'s weight/score to the new node.

```python
def compress(node_a: HFN, node_b: HFN, forest: Forest, compressed_id: str) -> HFN
```
Creates a compressed node from co-occurring pair. Registers it in Forest (does not deregister originals — Observer may keep them). Returns new node.

---

## 5. Observer Changes

Observer shrinks significantly. Changes:

- **Remove:** `_local_crowding`, `_local_density_ratio`, `_nearest_prior_dist`, `_build_prior_mus`, `_accuracy`. Move to Evaluator.
- **Remove:** direct `node_a.recombine(node_b)` + Forest register/deregister calls in `_check_absorption` and `_check_compression_candidates`. Replace with `recombination.absorb()` / `recombination.compress()` calls.
- **Keep:** `_update_weights`, `_update_scores`, `_track_cooccurrence`, `_expand`, `_check_node_creation`. These are Observer's core responsibilities.
- **Keep:** all HPM strategy flags — they are Observer configuration.
- **Add:** `evaluator: Evaluator` and `recombination: Recombination` as collaborators, constructed at `__init__` time or injected.

`_update_scores` delegates score formula to `evaluator.score()`.
`_check_absorption` delegates geometry queries to `evaluator.hausdorff_candidates()`, `evaluator.crowding()`, `evaluator.persistence_scores()`, then calls `recombination.absorb()` for confirmed candidates.
`_check_compression_candidates` delegates ranking to `evaluator.nearest_prior_dist()`, then calls `recombination.compress()`.
`_check_residual_surprise` delegates density check to `evaluator.density_ratio()`, and may also query `evaluator.coverage_gap()` to calibrate creation threshold.

---

## 6. File Changes

| File | Change |
|------|--------|
| `hfn/evaluator.py` | New — all evaluation methods |
| `hfn/recombination.py` | New — absorb + compress operations |
| `hfn/observer.py` | Refactor — remove moved methods, add Evaluator + Recombination collaborators |
| `hfn/__init__.py` | Export Evaluator, Recombination |
| `tests/hfn/test_evaluator.py` | New — unit tests for each Evaluator method |
| `tests/hfn/test_recombination.py` | New — unit tests for absorb + compress |
| `tests/hfn/test_observer.py` | Update — existing Observer tests should pass unchanged |

---

## 7. What Does Not Change

- HFN, Forest, TieredForest — no changes
- Observer public interface (`observe()`, `register()`, `get_weight()`, `get_score()`) — no changes
- All existing tests — must pass without modification
- NLP experiment — no changes needed (uses Observer public interface only)

---

## 8. HPM Framework Alignment

After this refactor:

- **HPM layer 2 (Pattern dynamics):** Observer — weight/score updates, co-occurrence tracking
- **HPM layer 3 (Pattern evaluators/gatekeepers):** Evaluator — coherence, utility, curiosity, boredom, reinforcement hooks
- **Structural execution:** Recombination — absorption, compression

The gap detection methods (`coverage_gap`, `underrepresented_regions`) provide the Observer with meta-awareness of its knowledge boundaries. This is the precondition for curiosity-driven learning and later integration with external substrates/fields: Observer queries Evaluator for gaps, and those gap signals can be forwarded to a field/substrate requesting new observations.

---

## 9. Extension Points

- `reinforcement_signal` is designed to be subclassed or injected — connects external reward models, social context evaluators, or field signals without modifying Observer or Evaluator internals.
- `underrepresented_regions` returns gap mu-vectors that Observer can expose upward — the interface for "I need more observations here" is left to the HPM AI layer above Observer.
- `curiosity` and `boredom` signals are currently unused by Observer decision logic — they are present as evaluations Observer can act on in future passes.
