"""
MetaObserver — L5 pattern layer for the HFN system.

Implements the HPM insight that meta-patterns are structurally identical to
any other pattern: the same Observer/Forest machinery, applied to a vector
encoding the *learning signal* of a lower-level Observer rather than raw data.

Usage:
    meta = MetaObserver(cold_dir=Path("run/meta"))
    # After each lower-level observation step:
    meta_result = meta.step(result, obs)
    # meta_result is itself an ExplanationResult — composable upward.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfn.observer import Observer, ExplanationResult

# Dimensionality of the meta-observation vector
META_D = 12


def observer_state_to_vec(result: ExplanationResult, observer: Observer) -> np.ndarray:
    """
    Convert a lower-level Observer's learning signal into a META_D-dimensional
    vector suitable for observation by a MetaObserver.

    All values are normalized to approximately [0, 1].

    Vector layout:
      [0]  residual_surprise          — how surprising the last observation was
      [1]  mean accuracy              — mean accuracy across explaining nodes
      [2]  mean weight                — mean node weight in forest
      [3]  weight std                 — spread of weights (diversity signal)
      [4]  min weight                 — weakest node
      [5]  max weight                 — strongest node
      [6]  mean score                 — mean score across nodes
      [7]  forest size (log-scaled)   — log1p(N) / 7.0
      [8]  leaf fraction              — proportion of leaf nodes
      [9]  explaining tree size       — len(explanation_tree) / 10.0
      [10] surprising leaves count    — len(surprising_leaves) / 5.0
      [11] mean miss count            — mean miss count / 10.0
    """
    vec = np.zeros(META_D)

    # [0] residual surprise — clip to [0, 1] range
    vec[0] = float(np.clip(result.residual_surprise / 10.0, 0.0, 1.0))

    # [1] mean accuracy of explaining nodes
    if result.accuracy_scores:
        vec[1] = float(np.clip(np.mean(list(result.accuracy_scores.values())), 0.0, 1.0))

    # [2–5] weight statistics
    weights = list(observer._weights.values())
    if weights:
        vec[2] = float(np.clip(np.mean(weights), 0.0, 1.0))
        vec[3] = float(np.clip(np.std(weights), 0.0, 1.0))
        vec[4] = float(np.clip(min(weights), 0.0, 1.0))
        vec[5] = float(np.clip(max(weights), 0.0, 1.0))

    # [6] mean score
    scores = list(observer._scores.values())
    if scores:
        vec[6] = float(np.clip((np.mean(scores) + 1.0) / 2.0, 0.0, 1.0))

    # [7] forest size (log-scaled, normalize assuming ~1000 nodes max)
    n_nodes = len(observer.forest)
    vec[7] = float(np.clip(np.log1p(n_nodes) / 7.0, 0.0, 1.0))

    # [8] leaf fraction
    if n_nodes > 0:
        nodes = observer.forest.active_nodes()
        n_leaves = sum(1 for n in nodes if not n.children())
        vec[8] = float(n_leaves / n_nodes)

    # [9] explaining tree size
    vec[9] = float(np.clip(len(result.explanation_tree) / 10.0, 0.0, 1.0))

    # [10] surprising leaves count
    vec[10] = float(np.clip(len(result.surprising_leaves) / 5.0, 0.0, 1.0))

    # [11] mean miss count
    miss_counts = list(observer._miss_counts.values())
    if miss_counts:
        vec[11] = float(np.clip(np.mean(miss_counts) / 10.0, 0.0, 1.0))

    return vec


class MetaObserver:
    """
    An Observer that watches the learning signal of another Observer.

    Internally uses the same TieredForest + Observer machinery as any other
    level. The only difference is what it observes: meta-vectors from
    observer_state_to_vec() rather than domain data.

    Composable: step() returns an ExplanationResult, so a MetaObserver can
    itself be observed by a higher MetaObserver.
    """

    def __init__(self, cold_dir: Path, hot_cap: int = 50):
        from hfn.tiered_forest import TieredForest
        from hfn.observer import Observer
        cold_dir = Path(cold_dir)
        cold_dir.mkdir(parents=True, exist_ok=True)
        self._forest = TieredForest(D=META_D, cold_dir=cold_dir, hot_cap=hot_cap)
        self.observer = Observer(forest=self._forest)

    def step(self, result: ExplanationResult, source_observer: Observer) -> ExplanationResult:
        """
        Observe one meta-step: convert the source observer's learning signal
        to a vector and pass it through the internal Observer.

        Returns the MetaObserver's own ExplanationResult, composable upward.
        """
        vec = observer_state_to_vec(result, source_observer)
        return self.observer.observe(vec)

    def save_state(self, path: Path) -> None:
        """Persist internal observer weights/scores."""
        self.observer.save_state(path)

    def load_state(self, path: Path) -> None:
        """Restore internal observer weights/scores."""
        self.observer.load_state(path)

    def prune(self, min_weight: float = 1e-4) -> int:
        """Remove low-weight leaf nodes from the meta-forest."""
        return self.observer.prune(min_weight)
