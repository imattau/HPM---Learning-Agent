# Observer Refactor: Evaluator + Recombination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract Evaluator (pure evaluation service) and Recombination (structural executor) from Observer, shrinking Observer to a coordinator that tracks state and delegates evaluation/execution.

**Architecture:** Observer keeps all mutable state (weights, scores, co-occurrence, miss counts) and all decision logic. Evaluator is a stateless service with three responsibility classes: fractal geometry evaluations, gap/unknown detection, HPM framework evaluations. Recombination executes absorb/compress operations against the Forest and returns new HFN nodes.

**Tech Stack:** Python, numpy, hfn library (hfn.hfn, hfn.forest, hfn.fractal)

**Spec:** `docs/superpowers/specs/2026-03-29-observer-refactor-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `hfn/evaluator.py` | Create | All evaluation methods — fractal geometry, gap detection, HPM framework |
| `hfn/recombination.py` | Create | absorb() and compress() structural operations |
| `hfn/observer.py` | Modify | Remove moved methods; add Evaluator + Recombination collaborators |
| `hfn/__init__.py` | Modify | Export Evaluator, Recombination |
| `tests/hfn/test_evaluator.py` | Create | Unit tests for all Evaluator methods |
| `tests/hfn/test_recombination.py` | Create | Unit tests for absorb + compress |
| `tests/hpm_fractal_node/test_observer.py` | Verify | Must pass unchanged after Observer refactor |

---

## Task 1: Evaluator

**Files:**
- Create: `hfn/evaluator.py`
- Create: `tests/hfn/test_evaluator.py`

### Background

Evaluator is a pure evaluation service — stateless, no Forest mutation. It takes nodes and scalar inputs, returns numeric results. Three responsibility classes:
1. **Fractal geometry**: crowding, density_ratio, nearest_prior_dist, hausdorff_candidates, persistence_scores
2. **Gap detection**: coverage_gap, underrepresented_regions
3. **HPM framework**: accuracy, description_length, score, coherence, curiosity, boredom, reinforcement_signal

`reinforcement_signal` defaults to 0.0 and is designed for subclassing.

- [ ] **Step 1: Write failing tests**

```python
# tests/hfn/test_evaluator.py
import numpy as np
import pytest
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.evaluator import Evaluator


def _node(mu, id="n"):
    D = len(mu)
    return HFN(mu=np.array(mu, dtype=float), sigma=np.eye(D), id=id)


@pytest.fixture
def ev():
    return Evaluator()


# --- Fractal geometry ---

def test_crowding_counts_nearby(ev):
    nodes = [_node([0.0], "a"), _node([0.1], "b"), _node([5.0], "c")]
    # a and b are within radius=0.5 of mu=[0.0]; c is not
    count = ev.crowding(np.array([0.0]), nodes, radius=0.5)
    assert count == 2  # a and b


def test_crowding_skips_protected(ev):
    nodes = [_node([0.0], "a"), _node([0.1], "b")]
    count = ev.crowding(np.array([0.0]), nodes, radius=0.5, protected_ids={"a"})
    assert count == 1  # only b


def test_density_ratio_sparse(ev):
    # Only 3 nodes far apart — querying a distant point should be low density
    nodes = [_node([0.0], "a"), _node([10.0], "b"), _node([20.0], "c")]
    ratio = ev.density_ratio(np.array([100.0]), nodes, radius=1.0)
    assert ratio == 0.0  # no nodes near x=100


def test_density_ratio_dense(ev):
    nodes = [_node([0.0], "a"), _node([0.1], "b"), _node([0.2], "c")]
    ratio = ev.density_ratio(np.array([0.1]), nodes, radius=1.0)
    assert ratio > 1.0  # denser than average


def test_nearest_prior_dist(ev):
    prior_mus = np.array([[0.0, 0.0], [5.0, 5.0]])
    dist = ev.nearest_prior_dist(np.array([1.0, 0.0]), prior_mus)
    assert abs(dist - 1.0) < 1e-9


def test_nearest_prior_dist_none(ev):
    assert ev.nearest_prior_dist(np.array([1.0]), None) == float("inf")


def test_hausdorff_candidates(ev):
    D = 2
    low_w = _node([0.0, 0.0], "low")
    high_w = _node([0.1, 0.0], "high")
    nodes = [low_w, high_w]
    weights = {"low": 0.05, "high": 0.8}
    candidates = ev.hausdorff_candidates(nodes, weights, threshold=0.5, weight_floor=0.2)
    assert len(candidates) == 1
    assert candidates[0][0].id == "low"
    assert candidates[0][1].id == "high"


def test_hausdorff_candidates_protected_skipped(ev):
    low_w = _node([0.0, 0.0], "low")
    high_w = _node([0.1, 0.0], "high")
    nodes = [low_w, high_w]
    weights = {"low": 0.05, "high": 0.8}
    candidates = ev.hausdorff_candidates(
        nodes, weights, threshold=0.5, weight_floor=0.2,
        protected_ids={"low"}
    )
    assert candidates == []


def test_persistence_scores_returns_dict(ev):
    nodes = [_node([0.0, 0.0], "a"), _node([1.0, 0.0], "b")]
    weights = {"a": 0.5, "b": 0.3}
    scores = ev.persistence_scores(nodes, weights)
    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"a", "b"}


# --- Gap detection ---

def test_coverage_gap_no_nodes(ev):
    gap = ev.coverage_gap(np.array([0.0]), [], radius=1.0)
    assert gap == 1.0


def test_coverage_gap_nearby_node(ev):
    nodes = [_node([0.1], "a"), _node([0.2], "b"), _node([0.3], "c")]
    gap_near = ev.coverage_gap(np.array([0.15]), nodes, radius=1.0)
    gap_far = ev.coverage_gap(np.array([100.0]), nodes, radius=1.0)
    assert gap_near < gap_far


def test_underrepresented_regions_sparse(ev):
    # Cluster at 0, isolated node at 100 — region around 100 is underrepresented
    nodes = [
        _node([0.0], "a"), _node([0.1], "b"), _node([0.2], "c"),
        _node([100.0], "d"),
    ]
    gaps = ev.underrepresented_regions(nodes)
    # The isolated node at 100 should be in a gap region
    assert len(gaps) > 0


def test_underrepresented_regions_too_few_nodes(ev):
    nodes = [_node([0.0], "a"), _node([1.0], "b")]
    gaps = ev.underrepresented_regions(nodes)
    assert gaps == []


# --- HPM framework evaluations ---

def test_accuracy_exact_match(ev):
    node = _node([0.0, 0.0])
    x = np.array([0.0, 0.0])
    acc = ev.accuracy(x, node)
    assert 0.0 < acc <= 1.0


def test_accuracy_far_point(ev):
    node = _node([0.0, 0.0])
    x_near = np.array([0.0, 0.0])
    x_far = np.array([100.0, 100.0])
    assert ev.accuracy(x_near, node) > ev.accuracy(x_far, node)


def test_description_length(ev):
    node = _node([0.0, 0.0])
    dl = ev.description_length(node)
    assert dl > 0.0


def test_score(ev):
    node = _node([0.0, 0.0])
    x = np.array([0.0, 0.0])
    s = ev.score(x, node, lambda_complexity=0.1)
    acc = ev.accuracy(x, node)
    dl = ev.description_length(node)
    assert abs(s - (acc - 0.1 * dl)) < 1e-9


def test_coherence_range(ev):
    node = _node([0.0, 0.0])
    c = ev.coherence(node)
    assert 0.0 <= c <= 1.0


def test_curiosity_no_nodes(ev):
    c = ev.curiosity(np.array([0.0]), [], {})
    assert c == 1.0


def test_curiosity_well_covered(ev):
    nodes = [_node([0.0], "a"), _node([0.1], "b")]
    weights = {"a": 0.9, "b": 0.8}
    c_near = ev.curiosity(np.array([0.05]), nodes, weights)
    c_far = ev.curiosity(np.array([100.0]), nodes, weights)
    assert c_far > c_near


def test_boredom_high_weight_low_score(ev):
    b = ev.boredom("n", weights={"n": 0.9}, scores={"n": -2.0})
    assert b > 0.0


def test_boredom_low_weight(ev):
    b = ev.boredom("n", weights={"n": 0.1}, scores={"n": -2.0})
    assert b < ev.boredom("n", weights={"n": 0.9}, scores={"n": -2.0})


def test_reinforcement_signal_default(ev):
    assert ev.reinforcement_signal("any_id") == 0.0


def test_reinforcement_signal_subclass():
    class RewardEvaluator(Evaluator):
        def reinforcement_signal(self, node_id):
            return 1.0 if node_id == "good" else 0.0

    rev = RewardEvaluator()
    assert rev.reinforcement_signal("good") == 1.0
    assert rev.reinforcement_signal("other") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_evaluator.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'hfn.evaluator'`

- [ ] **Step 3: Implement hfn/evaluator.py**

```python
"""
HPM Evaluator — pure evaluation service for the Observer.

The Evaluator has no state and does not modify the Forest. It takes nodes
and Observer state as inputs and returns numeric results.

Three responsibility classes:
  1. Fractal geometry — crowding, density, persistence, Hausdorff candidates
  2. Gap detection    — what the Observer does not yet know
  3. HPM framework    — utility, coherence, curiosity, boredom, reinforcement
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfn.hfn import HFN


class Evaluator:
    """
    Pure evaluation service. Stateless — construct once and reuse.

    All methods are safe to call concurrently. No Forest mutations.
    Subclass and override reinforcement_signal() to connect external
    reward models, social context evaluators, or field signals.
    """

    # ------------------------------------------------------------------
    # 1. Fractal geometry evaluations
    # ------------------------------------------------------------------

    def crowding(
        self,
        node_mu: np.ndarray,
        nodes,
        radius: float,
        protected_ids: set[str] | None = None,
    ) -> int:
        """
        Count non-protected nodes within radius of node_mu.

        Used by Observer for multifractal-guided absorption: high crowding
        means the node is in a hot spot and should be absorbed sooner.
        """
        protected_ids = protected_ids or set()
        count = 0
        for other in nodes:
            if other.id in protected_ids:
                continue
            if float(np.linalg.norm(other.mu - node_mu)) < radius:
                count += 1
        return count

    def density_ratio(
        self,
        x: np.ndarray,
        nodes,
        radius: float,
    ) -> float:
        """
        Ratio of local node density at x to mean density across all nodes.

        Used by Observer for lacunarity-guided creation suppression: if ratio
        exceeds lacunarity_creation_factor, the region is dense and a new node
        would be redundant.

        Returns 0.0 if fewer than 3 nodes exist (always allow creation).
        """
        node_list = list(nodes)
        if len(node_list) < 3:
            return 0.0
        mus = np.array([n.mu for n in node_list])
        dists_to_x = np.linalg.norm(mus - x, axis=1)
        local_count = float(np.sum(dists_to_x < radius))
        mean_nn = float(np.mean([
            np.sort(np.linalg.norm(mus - mus[i], axis=1))[1]
            for i in range(len(mus))
        ]))
        expected_local = radius / (mean_nn + 1e-9)
        return local_count / (expected_local + 1e-9)

    def nearest_prior_dist(
        self,
        mu: np.ndarray,
        prior_mus: np.ndarray | None,
    ) -> float:
        """
        Euclidean distance from mu to the nearest prior node in mu-space.

        prior_mus: stacked matrix of prior node mu-vectors (N x D), or None.
        Used to rank compression candidates by proximity to existing priors.
        """
        if prior_mus is None or len(prior_mus) == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(prior_mus - mu, axis=1)))

    def hausdorff_candidates(
        self,
        nodes,
        weights: dict[str, float],
        threshold: float,
        weight_floor: float,
        protected_ids: set[str] | None = None,
    ) -> list[tuple["HFN", "HFN"]]:
        """
        Return (absorbed, dominant) pairs eligible for geometric absorption.

        A node is a candidate if:
        - It is not protected
        - Its weight is below weight_floor
        - There exists a better-weighted non-protected node within threshold distance

        Observer decides whether to absorb each returned pair.
        """
        protected_ids = protected_ids or set()
        candidates = []
        node_list = list(nodes)
        for node in node_list:
            if node.id in protected_ids:
                continue
            if weights.get(node.id, 0.0) >= weight_floor:
                continue
            for other in node_list:
                if other.id == node.id or other.id in protected_ids:
                    continue
                if weights.get(other.id, 0.0) <= weights.get(node.id, 0.0):
                    continue
                dist = float(np.linalg.norm(node.mu - other.mu))
                if dist < threshold:
                    candidates.append((node, other))
                    break
        return candidates

    def persistence_scores(
        self,
        nodes,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Return persistence score per node id.

        Thin wrapper over hfn.fractal.persistence_scores. High persistence
        means the node has been consistently useful; Observer scales absorption
        miss threshold up for persistent nodes.
        """
        from hfn.fractal import persistence_scores as _ps
        return _ps(list(nodes), weights)

    # ------------------------------------------------------------------
    # 2. Gap / unknown detection
    # ------------------------------------------------------------------

    def coverage_gap(
        self,
        x: np.ndarray,
        nodes,
        radius: float,
    ) -> float:
        """
        How sparse is the region around observation x given current nodes.

        Returns a value in (0, 1]: 1.0 = completely unknown region (no nodes),
        approaching 0 as more nodes exist near x.

        Observer uses this to calibrate creation threshold: high gap → raise
        readiness to create a new node. Future: expose to external substrates
        as "I need more observations here."
        """
        node_list = list(nodes)
        if not node_list:
            return 1.0
        mus = np.array([n.mu for n in node_list])
        dists = np.linalg.norm(mus - x, axis=1)
        nearby = float(np.sum(dists < radius))
        return 1.0 / (1.0 + nearby)

    def underrepresented_regions(
        self,
        nodes,
    ) -> list[np.ndarray]:
        """
        Regions of observation space with low node density.

        Uses nearest-neighbour distance to identify nodes sitting in sparse
        areas (their NN distance exceeds mean + 1 std). Returns representative
        mu-vectors for each gap region.

        Observer can expose these as "I need more observations here" signals
        to external substrates/fields.

        Returns empty list if fewer than 3 nodes exist.
        """
        node_list = list(nodes)
        if len(node_list) < 3:
            return []
        mus = np.array([n.mu for n in node_list])
        nn_dists = []
        for i in range(len(mus)):
            dists = np.linalg.norm(mus - mus[i], axis=1)
            dists[i] = np.inf
            nn_dists.append(float(np.min(dists)))
        nn_array = np.array(nn_dists)
        threshold = float(np.mean(nn_array) + np.std(nn_array))
        return [mus[i] for i in range(len(mus)) if nn_array[i] > threshold]

    # ------------------------------------------------------------------
    # 3. HPM framework evaluations
    # ------------------------------------------------------------------

    def accuracy(self, x: np.ndarray, node: "HFN") -> float:
        """
        Functional utility of node for observation x.

        Normalised log probability: higher = better fit.
        Range: (0, 1].
        """
        lp = node.log_prob(x)
        return float(1.0 / (1.0 + abs(lp)))

    def description_length(self, node: "HFN") -> float:
        """
        Complexity proxy for node. Wraps node.description_length()."""
        return float(node.description_length())

    def score(
        self,
        x: np.ndarray,
        node: "HFN",
        lambda_complexity: float,
    ) -> float:
        """
        Combined utility score: accuracy - lambda * description_length.

        Higher = more useful, less complex. Used by Observer._update_scores.
        """
        return self.accuracy(x, node) - lambda_complexity * self.description_length(node)

    def coherence(self, node: "HFN") -> float:
        """
        Internal consistency of node's pattern, based on sigma matrix.

        Low condition number = tight, consistent covariance = high coherence.
        Returns value in [0, 1].
        """
        try:
            cond = float(np.linalg.cond(node.sigma))
            return float(1.0 / (1.0 + np.log1p(max(0.0, cond - 1.0))))
        except Exception:
            return 0.0

    def curiosity(
        self,
        x: np.ndarray,
        nodes,
        weights: dict[str, float],
    ) -> float:
        """
        Novelty signal for observation x.

        How far is x from well-weighted nodes (weight > 0.5)?
        High value = Observer has not seen patterns like this.
        Returns 1.0 if no well-weighted nodes exist.
        """
        well_weighted = [n for n in nodes if weights.get(n.id, 0.0) > 0.5]
        if not well_weighted:
            return 1.0
        mus = np.array([n.mu for n in well_weighted])
        dists = np.linalg.norm(mus - x, axis=1)
        min_dist = float(np.min(dists))
        # Sigmoid-like: 0.5 at dist=1.0, approaching 1 for large dist
        return float(1.0 / (1.0 + np.exp(-(min_dist - 1.0))))

    def boredom(
        self,
        node_id: str,
        weights: dict[str, float],
        scores: dict[str, float],
    ) -> float:
        """
        Redundancy signal for a node.

        High weight + low (negative) score = node is over-represented and
        no longer improving. Observer can use this to flag stale nodes.
        Returns 0.0 for nodes with positive score.
        """
        w = weights.get(node_id, 0.0)
        s = scores.get(node_id, 0.0)
        return float(w * max(0.0, -s))

    def reinforcement_signal(self, node_id: str) -> float:
        """
        External reinforcement signal for node_id.

        Default: 0.0 (no external signal).

        Subclass Evaluator and override this method to connect external
        reward models, social context evaluators, or field/substrate signals.
        The Observer uses this in weight update decisions.
        """
        return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_evaluator.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add hfn/evaluator.py tests/hfn/test_evaluator.py
git commit -m "feat: add Evaluator — fractal geometry, gap detection, HPM framework evaluations"
```

---

## Task 2: Recombination

**Files:**
- Create: `hfn/recombination.py`
- Create: `tests/hfn/test_recombination.py`

### Background

Recombination is a pure structural executor. It performs `absorb` and `compress` operations against the Forest and returns new HFN nodes. No decision logic — Observer decides when; Recombination executes how.

`absorb(absorbed, dominant, forest)`: dominant is the surviving node. Both are deregistered, merged node is registered.

`compress(node_a, node_b, forest, compressed_id)`: creates compressed node, registers it. Does NOT deregister originals (Observer keeps them active).

The returned HFN from either method carries no weight/score state — Observer initialises those.

- [ ] **Step 1: Write failing tests**

```python
# tests/hfn/test_recombination.py
import numpy as np
import pytest
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.recombination import Recombination


def _node(mu, id="n"):
    D = len(mu)
    return HFN(mu=np.array(mu, dtype=float), sigma=np.eye(D), id=id)


def _forest(*nodes):
    D = len(nodes[0].mu)
    f = Forest(D=D)
    for n in nodes:
        f.register(n)
    return f


@pytest.fixture
def rec():
    return Recombination()


def test_absorb_registers_new_node(rec):
    absorbed = _node([0.0, 0.0], "absorbed")
    dominant = _node([1.0, 0.0], "dominant")
    forest = _forest(absorbed, dominant)

    new_node = rec.absorb(absorbed, dominant, forest)

    assert new_node.id in forest
    assert "absorbed" not in forest
    assert "dominant" not in forest


def test_absorb_returns_hfn(rec):
    absorbed = _node([0.0, 0.0], "absorbed")
    dominant = _node([1.0, 0.0], "dominant")
    forest = _forest(absorbed, dominant)
    new_node = rec.absorb(absorbed, dominant, forest)
    assert isinstance(new_node, HFN)


def test_absorb_mu_between_inputs(rec):
    absorbed = _node([0.0, 0.0], "absorbed")
    dominant = _node([2.0, 0.0], "dominant")
    forest = _forest(absorbed, dominant)
    new_node = rec.absorb(absorbed, dominant, forest)
    # Recombined mu should be between absorbed and dominant
    assert 0.0 <= new_node.mu[0] <= 2.0


def test_absorb_returns_valid_id(rec):
    absorbed = _node([0.0, 0.0], "absorbed")
    dominant = _node([1.0, 0.0], "dominant")
    forest = _forest(absorbed, dominant)
    new_node = rec.absorb(absorbed, dominant, forest)
    # New node has a non-empty string id (UUID from HFN.recombine)
    assert isinstance(new_node.id, str) and len(new_node.id) > 0


def test_compress_registers_new_node(rec):
    a = _node([0.0, 0.0], "aaa")
    b = _node([1.0, 0.0], "bbb")
    forest = _forest(a, b)
    compressed_id = "compressed(aaa,bbb)"

    new_node = rec.compress(a, b, forest, compressed_id)

    assert compressed_id in forest
    assert new_node.id == compressed_id


def test_compress_keeps_originals(rec):
    a = _node([0.0, 0.0], "aaa")
    b = _node([1.0, 0.0], "bbb")
    forest = _forest(a, b)

    rec.compress(a, b, forest, "compressed(aaa,bbb)")

    # Originals remain active
    assert "aaa" in forest
    assert "bbb" in forest


def test_compress_returns_hfn(rec):
    a = _node([0.0, 0.0], "aaa")
    b = _node([1.0, 0.0], "bbb")
    forest = _forest(a, b)
    new_node = rec.compress(a, b, forest, "compressed(aaa,bbb)")
    assert isinstance(new_node, HFN)


def test_compress_mu_between_inputs(rec):
    a = _node([0.0, 0.0], "aaa")
    b = _node([4.0, 0.0], "bbb")
    forest = _forest(a, b)
    new_node = rec.compress(a, b, forest, "compressed(aaa,bbb)")
    assert 0.0 <= new_node.mu[0] <= 4.0


def test_absorb_stale_node_no_crash(rec):
    # If dominant was already deregistered, Forest.deregister is idempotent
    absorbed = _node([0.0, 0.0], "absorbed")
    dominant = _node([1.0, 0.0], "dominant")
    forest = _forest(absorbed, dominant)
    forest.deregister("dominant")  # pre-deregister to simulate stale pair
    # Should not raise
    new_node = rec.absorb(absorbed, dominant, forest)
    assert new_node.id in forest
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_recombination.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'hfn.recombination'`

- [ ] **Step 3: Implement hfn/recombination.py**

```python
"""
HPM Recombination — structural executor for the Observer.

Performs absorb and compress operations against the Forest. No decision
logic — Observer decides when; Recombination executes how.

The returned HFN from either method carries no weight/score state.
Observer initialises those after the call.
"""

from __future__ import annotations

from hfn.hfn import HFN
from hfn.forest import Forest


class Recombination:
    """
    Pure structural executor. Stateless — construct once and reuse.

    absorb: merge two nodes, deregister both, register result.
    compress: create a compressed node from a co-occurring pair, register it
              (originals remain active).
    """

    def absorb(
        self,
        absorbed: HFN,
        dominant: HFN,
        forest: Forest,
    ) -> HFN:
        """
        Recombine absorbed into dominant, replacing both with the merged node.

        dominant is the surviving node (higher weight per Observer's decision).
        Both are deregistered from Forest; merged node is registered.

        Returns the new HFN. It carries no weight/score state — Observer
        calls _init_node() then copies weight/score from dominant.
        """
        new_node = dominant.recombine(absorbed)
        forest.deregister(dominant.id)
        forest.deregister(absorbed.id)
        forest.register(new_node)
        return new_node

    def compress(
        self,
        node_a: HFN,
        node_b: HFN,
        forest: Forest,
        compressed_id: str,
    ) -> HFN:
        """
        Create a compressed node from a co-occurring pair and register it.

        compressed_id is constructed by Observer before this call
        (format: "compressed({id_a[:8]},{id_b[:8]})").

        Originals (node_a, node_b) are NOT deregistered — they remain active.
        Returns the new HFN. Observer initialises its weight/score.
        """
        new_node = node_a.recombine(node_b)
        new_node.id = compressed_id  # type: ignore[misc]
        forest.register(new_node)
        return new_node
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_recombination.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add hfn/recombination.py tests/hfn/test_recombination.py
git commit -m "feat: add Recombination — absorb and compress structural operations"
```

---

## Task 3: Observer refactor

**Files:**
- Modify: `hfn/observer.py`

### Background

Remove the five methods that moved to Evaluator (`_local_crowding`, `_local_density_ratio`, `_nearest_prior_dist`, `_build_prior_mus`, `_accuracy`). Add `self.evaluator` and `self.recombination` collaborators. Update `_update_scores`, `_check_absorption`, `_check_compression_candidates`, `_check_residual_surprise` to delegate to them.

Key implementation notes:
- `_prior_mus` cache stays on Observer (it manages the cache); Observer builds the matrix inline when needed and passes it to `evaluator.nearest_prior_dist()`.
- `_check_absorption` uses a snapshot: `list(self.forest.active_nodes())` at the top of the method to avoid mid-pass invalidation.
- `_recurrence` (adaptive compression) stays on Observer.
- `_check_compression_candidates` must check `compressed_id in self.forest` before calling compress.
- After `recombination.absorb()`, Observer calls `self._init_node(new_node)` then copies weight/score from dominant.

The existing Observer tests (`tests/hpm_fractal_node/test_observer.py`) must pass without modification.

- [ ] **Step 1: Run existing Observer tests to establish baseline**

```bash
PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_observer.py -v
```
Note the count of passing tests. All must still pass after the refactor.

- [ ] **Step 2: Add imports and collaborators to Observer.__init__**

In `hfn/observer.py`, add at the top:
```python
from hfn.evaluator import Evaluator
from hfn.recombination import Recombination
```

In `Observer.__init__`, after setting all config attributes (around line 143), add:
```python
self.evaluator = Evaluator()
self.recombination = Recombination()
```

- [ ] **Step 3: Update _update_scores to use evaluator.score()**

Replace the existing `_update_scores` method:

```python
def _update_scores(self, result: ExplanationResult) -> None:
    explaining_ids = {n.id for n in result.explanation_tree}
    for node in self.forest.active_nodes():
        nid = node.id
        x_proxy = node.mu  # use mu as proxy when x not available
        if nid in explaining_ids:
            # Use the observation's accuracy, not the proxy
            acc = result.accuracy_scores.get(nid, 0.0)
            self._scores[nid] = acc - self.lambda_complexity * self.evaluator.description_length(node)
        else:
            self._scores[nid] = -self.lambda_complexity * self.evaluator.description_length(node)
```

Wait — the original `_update_scores` uses `acc` from `result.accuracy_scores`, not from evaluator. Keep this pattern:

```python
def _update_scores(self, result: ExplanationResult) -> None:
    explaining_ids = {n.id for n in result.explanation_tree}
    for node in self.forest.active_nodes():
        nid = node.id
        acc = result.accuracy_scores.get(nid, 0.0) if nid in explaining_ids else 0.0
        self._scores[nid] = acc - self.lambda_complexity * self.evaluator.description_length(node)
```

- [ ] **Step 4: Update _check_absorption to use snapshot + evaluator + recombination**

Replace the existing `_check_absorption` method with:

```python
def _check_absorption(self) -> None:
    # Snapshot: avoids mid-pass invalidation when nodes are deregistered
    snapshot = list(self.forest.active_nodes())

    # Compute persistence scores once per pass if needed
    p_scores: dict[str, float] = {}
    p_max: float = 1.0
    if self.persistence_guided_absorption:
        p_scores = self.evaluator.persistence_scores(snapshot, self._weights)
        p_max = max(
            (v for v in p_scores.values() if v != float("inf")),
            default=1.0,
        )

    # Geometric absorption pass (Hausdorff-based)
    if self.hausdorff_absorption_threshold is not None:
        candidates = self.evaluator.hausdorff_candidates(
            snapshot,
            self._weights,
            threshold=self.hausdorff_absorption_threshold,
            weight_floor=self.hausdorff_absorption_weight_floor,
            protected_ids=self.protected_ids,
        )
        absorbed_this_pass: set[str] = set()
        for absorbed_node, dominant_node in candidates:
            if absorbed_node.id in absorbed_this_pass or dominant_node.id in absorbed_this_pass:
                continue
            if self.forest.get(absorbed_node.id) is None or self.forest.get(dominant_node.id) is None:
                continue
            new_node = self.recombination.absorb(absorbed_node, dominant_node, self.forest)
            self._init_node(new_node)
            self._weights[new_node.id] = self._weights.get(dominant_node.id, self.w_init)
            self._scores[new_node.id] = self._scores.get(dominant_node.id, 0.0)
            self.absorbed_ids.add(absorbed_node.id)
            absorbed_this_pass.add(absorbed_node.id)
            absorbed_this_pass.add(dominant_node.id)

    # Miss-count absorption pass
    for node in snapshot:
        if node.id in self.protected_ids:
            continue
        if self.forest.get(node.id) is None:
            continue  # already absorbed in geometric pass

        effective_miss_threshold = self.absorption_miss_threshold
        if self.persistence_guided_absorption and p_max > 0:
            node_p = p_scores.get(node.id, 0.0)
            norm_p = min(node_p, p_max) / p_max
            effective_miss_threshold = int(
                round(self.absorption_miss_threshold * (1.0 + norm_p))
            )

        if self.multifractal_guided_absorption:
            crowding = self.evaluator.crowding(
                node.mu, snapshot, self.multifractal_crowding_radius,
                protected_ids=self.protected_ids,
            )
            if crowding >= self.multifractal_crowding_threshold:
                effective_miss_threshold = max(1, effective_miss_threshold // 2)

        if self._miss_counts.get(node.id, 0) < effective_miss_threshold:
            continue

        best_overlap = 0.0
        best_node = None
        for other in snapshot:
            if other.id == node.id or other.id in self.protected_ids:
                continue
            if self.forest.get(other.id) is None:
                continue
            kappa = node.overlap(other)
            if kappa > self.absorption_overlap_threshold and kappa > best_overlap:
                best_overlap = kappa
                best_node = other

        if best_node is None:
            continue

        new_node = self.recombination.absorb(node, best_node, self.forest)
        self._init_node(new_node)
        self._weights[new_node.id] = self._weights.get(best_node.id, self.w_init)
        self._scores[new_node.id] = self._scores.get(best_node.id, 0.0)
        self.absorbed_ids.add(node.id)
```

- [ ] **Step 5: Add _build_prior_mus_matrix helper and remove old private methods**

> **Do this before Step 6** — Step 6 calls `_build_prior_mus_matrix()`.

Add the new private helper to `Observer` (before the compression method):

```python
def _build_prior_mus_matrix(self) -> np.ndarray | None:
    """Stack mu vectors of all protected nodes into a matrix for evaluator calls."""
    prior_nodes = [
        n for n in self.forest.active_nodes() if n.id in self.protected_ids
    ]
    if not prior_nodes:
        return None
    return np.array([n.mu for n in prior_nodes])
```

Remove these five methods entirely from `observer.py` (they are now in Evaluator):
- `_build_prior_mus`
- `_nearest_prior_dist`
- `_local_density_ratio`
- `_local_crowding`
- `_accuracy`

Also remove `self._prior_mus: np.ndarray | None = None` from `__init__` (line ~163).

- [ ] **Step 6: Update _check_compression_candidates to use evaluator + recombination**

Replace the existing `_check_compression_candidates` method with:

```python
def _check_compression_candidates(self) -> None:
    threshold = self.compression_cooccurrence_threshold
    if self._recurrence is not None:
        threshold = self._recurrence.recommended_threshold(threshold)

    candidates: list[tuple[frozenset, list[str]]] = []
    for pair, count in list(self._cooccurrence.items()):
        if count < threshold:
            continue
        ids = list(pair)
        if ids[0] not in self.forest or ids[1] not in self.forest:
            continue
        compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
        if compressed_id in self.forest:
            continue
        candidates.append((pair, ids))

    if not candidates:
        return

    if self.recombination_strategy == "nearest_prior" and self.protected_ids:
        prior_mus = self._build_prior_mus_matrix()
        def _rank(item: tuple) -> float:
            _, ids = item
            a_mu = self.forest.get(ids[0]).mu
            b_mu = self.forest.get(ids[1]).mu
            midpoint = (a_mu + b_mu) / 2.0
            return self.evaluator.nearest_prior_dist(midpoint, prior_mus)
        candidates.sort(key=_rank)

    for pair, ids in candidates:
        node_a = self.forest.get(ids[0])
        node_b = self.forest.get(ids[1])
        if node_a is None or node_b is None:
            continue
        compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
        if compressed_id in self.forest:
            continue
        new_node = self.recombination.compress(node_a, node_b, self.forest, compressed_id)
        self.register(new_node)
        self._cooccurrence[pair] = 0
```

Note: `register()` calls both `forest.register()` and `_init_node()`. Since `compress()` already registers the node in the Forest, replace `self.register(new_node)` with just `self._init_node(new_node)`.

Corrected final lines:
```python
        new_node = self.recombination.compress(node_a, node_b, self.forest, compressed_id)
        self._init_node(new_node)
        self._cooccurrence[pair] = 0
```

- [ ] **Step 7: Update _check_residual_surprise to use evaluator.density_ratio()**

Replace the lacunarity check inside `_check_residual_surprise`:

```python
def _check_residual_surprise(self, x: np.ndarray, result: ExplanationResult) -> None:
    if len(self.forest) == 0 or result.residual_surprise >= self.residual_surprise_threshold:
        if self.lacunarity_guided_creation and len(self.forest) > 0:
            ratio = self.evaluator.density_ratio(
                x, self.forest.active_nodes(), self.lacunarity_creation_radius
            )
            if ratio > self.lacunarity_creation_factor:
                return
        D = x.shape[0]
        new_node = HFN(
            mu=x.copy(),
            sigma=np.eye(D),
            id=f"leaf_{len(self.forest)}",
        )
        self.register(new_node)
```

- [ ] **Step 8: Run all tests to verify Observer refactor passes**

```bash
PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_observer.py tests/hfn/ -v
```
Expected: all tests pass (same count as baseline from Step 1, plus new evaluator/recombination tests).

- [ ] **Step 9: Commit**

```bash
git add hfn/observer.py
git commit -m "refactor: Observer delegates to Evaluator + Recombination"
```

---

## Task 4: Exports + Full Verification

**Files:**
- Modify: `hfn/__init__.py`
- Verify: full test suite

- [ ] **Step 1: Add Evaluator and Recombination to hfn/__init__.py**

In `hfn/__init__.py`, add imports:
```python
from hfn.evaluator import Evaluator
from hfn.recombination import Recombination
```

Update the Public API docstring to include:
```
Evaluator    — pure evaluation service (fractal geometry, gap detection, HPM evaluations)
Recombination — structural executor (absorb, compress)
```

Add to `__all__`:
```python
"Evaluator", "Recombination",
```

- [ ] **Step 2: Run full test suite**

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
Expected: all tests pass. Note total count.

- [ ] **Step 3: Run NLP experiment to verify end-to-end**

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py
```
Expected: 3 passes, 100% explained, similar purity scores to pre-refactor run.

- [ ] **Step 4: Commit**

```bash
git add hfn/__init__.py
git commit -m "feat: export Evaluator and Recombination from hfn package"
```
