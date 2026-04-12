"""
Pattern Density Tracking (HPM Appendix A.8)

Implements D(h) = α·C(h) + β·E(h) + γ·F(h) where:

- C(h): structural connectivity (children, inputs, edges)
- E(h): evaluator reinforcement (weight + success history)
- F(h): field amplification (recency‑weighted usage)

All state is stored in dedicated TieredForest instances as HFN nodes,
maintaining fractal uniformity.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest

if TYPE_CHECKING:
    from hfn.observer import Observer


class PatternDensityTracker:
    """
    Tracks pattern density for an Observer. All state is stored as HFN nodes.
    """

    DENSITY_DIM = 4      # mu = [C, E, F, total]
    USAGE_DIM = 4        # mu = [latest_timestamp, count, recency_sum, _]

    def __init__(self, observer: Observer, cold_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        observer : Observer
            The Observer whose patterns are being tracked.
        cold_dir : Path, optional
            Directory for persistent storage. If None, a temporary directory is used.
        """
        self.observer = observer
        if cold_dir is None:
            cold_dir = Path(tempfile.mkdtemp(prefix="hfn_density_"))
        self._forest = TieredForest(
            D=self.DENSITY_DIM,
            cold_dir=cold_dir / "density",
            forest_id="density_tracker"
        )
        self._usage_forest = TieredForest(
            D=self.USAGE_DIM,
            cold_dir=cold_dir / "usage",
            forest_id="usage_tracker"
        )
        # Initialise density nodes for all existing patterns
        for node in observer.forest.active_nodes():
            self._init_density_node(node.id)

    # ------------------------------------------------------------------
    # Internal node management
    # ------------------------------------------------------------------

    def _init_density_node(self, pattern_id: str) -> HFN:
        node_id = f"density:{pattern_id}"
        if node_id not in self._forest:
            node = HFN(
                mu=np.zeros(self.DENSITY_DIM),
                sigma=np.ones(self.DENSITY_DIM),
                id=node_id,
                use_diag=True
            )
            self._forest.register(node)
            return node
        return self._forest.get(node_id)

    def _get_density_node(self, pattern_id: str) -> Optional[HFN]:
        return self._forest.get(f"density:{pattern_id}")

    def _get_usage_node(self, pattern_id: str) -> Optional[HFN]:
        return self._usage_forest.get(f"usage:{pattern_id}")

    def _init_usage_node(self, pattern_id: str, timestamp: float) -> HFN:
        node_id = f"usage:{pattern_id}"
        node = HFN(
            mu=np.array([timestamp, 1.0, 1.0, 0.0]),
            sigma=np.ones(self.USAGE_DIM),
            id=node_id,
            use_diag=True
        )
        self._usage_forest.register(node)
        return node

    def _recompute_total(self, dnode: HFN, alpha: float = 0.4, beta: float = 0.35, gamma: float = 0.25) -> None:
        """Recompute total density D = α·C + β·E + γ·F."""
        total = alpha * dnode.mu[0] + beta * dnode.mu[1] + gamma * dnode.mu[2]
        dnode.mu[3] = total

    # ------------------------------------------------------------------
    # Public update methods – called by Observer
    # ------------------------------------------------------------------

    def update_structural_connectivity(self, node: HFN) -> None:
        """Update C(h) from node's children, inputs, and edges."""
        dnode = self._get_density_node(node.id)
        if dnode is None:
            dnode = self._init_density_node(node.id)

        n_children = len(node.children())
        n_inputs = len(node.inputs) if node.inputs else 0
        n_edges = len(node.edges())
        # Heuristic: base connectivity of 0.1 for existing, max when ~10 children/inputs or ~20 edges
        c_val = min(1.0, 0.1 + (n_children + n_inputs) / 10.0 + (n_edges / 20.0))
        dnode.mu[0] = c_val
        self._recompute_total(dnode)

    def update_evaluator_reinforcement(self, pattern_id: str, success: bool) -> None:
        """Update E(h) using Observer's weight and success outcome."""
        dnode = self._get_density_node(pattern_id)
        if dnode is None:
            dnode = self._init_density_node(pattern_id)

        weight = self.observer.get_weight(pattern_id)
        current_e = dnode.mu[1]
        # Exponential moving average
        new_e = 0.7 * current_e + 0.3 * weight
        if success:
            new_e = min(1.0, new_e + 0.05)
        dnode.mu[1] = new_e
        self._recompute_total(dnode)

    def update_field_amplification(self, pattern_id: str, timestamp: float) -> None:
        """Update F(h) using recency‑weighted usage frequency."""
        dnode = self._get_density_node(pattern_id)
        if dnode is None:
            dnode = self._init_density_node(pattern_id)

        unode = self._get_usage_node(pattern_id)
        if unode is None:
            unode = self._init_usage_node(pattern_id, timestamp)
            field_amp = 1.0
        else:
            count = unode.mu[1] + 1
            recency_sum = unode.mu[2] * 0.9 + 1.0  # decay factor 0.9
            unode.mu[0] = timestamp
            unode.mu[1] = count
            unode.mu[2] = recency_sum
            # Normalise: assume max recency_sum ~10 for full amplification
            field_amp = min(1.0, recency_sum / 10.0)

        dnode.mu[2] = field_amp
        self._recompute_total(dnode)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_density(self, pattern_id: str) -> Optional[Tuple[float, float, float, float]]:
        """Return (C, E, F, total) for a pattern, or None if not tracked."""
        dnode = self._get_density_node(pattern_id)
        if dnode is None:
            return None
        return (float(dnode.mu[0]), float(dnode.mu[1]), float(dnode.mu[2]), float(dnode.mu[3]))

    def get_total_density(self, pattern_id: str) -> float:
        """Return total density D(h) for a pattern, or 0.0 if not tracked."""
        d = self.get_density(pattern_id)
        return d[3] if d else 0.0

    def should_prune(self, pattern_id: str, threshold: float = 0.3) -> bool:
        """
        HPM prediction: low‑density patterns with high epistemic loss are pruned.

        Epistemic loss is approximated as 1 - (0.5*weight + 0.5*score).
        """
        dens = self.get_density(pattern_id)
        if dens is None:
            return False
        total = dens[3]
        weight = self.observer.get_weight(pattern_id)
        score = self.observer.get_score(pattern_id)
        epistemic_loss = 1.0 - (0.5 * weight + 0.5 * max(0.0, min(1.0, score)))
        return total < threshold and epistemic_loss > 0.7

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, top_n: int = 10) -> str:
        """Return a human‑readable density report."""
        lines = ["Pattern Density Report (top {} by total density):".format(top_n)]
        items = []
        for node in self._forest.active_nodes():
            if node.id.startswith("density:"):
                pat_id = node.id[len("density:"):]
                items.append((pat_id, tuple(node.mu)))
        items.sort(key=lambda x: x[1][3], reverse=True)
        for pat_id, (c, e, f, tot) in items[:top_n]:
            lines.append(f"  {pat_id[:20]}: C={c:.2f} E={e:.2f} F={f:.2f} D={tot:.2f}")
        if len(items) > top_n:
            lines.append(f"  ... and {len(items)-top_n} more")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Observer integration helper
# ------------------------------------------------------------------

def attach_density_tracker(observer, cold_dir: Optional[Path] = None) -> PatternDensityTracker:
    """
    Attach a PatternDensityTracker to an existing Observer.

    This sets observer.density_tracker and returns it.
    """
    tracker = PatternDensityTracker(observer, cold_dir)
    observer.density_tracker = tracker
    return tracker
