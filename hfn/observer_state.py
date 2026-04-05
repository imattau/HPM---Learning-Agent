"""
Typed meta-state accessors for Observer.
"""
from __future__ import annotations

from dataclasses import dataclass

from hfn.tiered_forest import TieredForest


@dataclass(frozen=True)
class NodeStateSnapshot:
    weight: float
    score: float
    miss_count: int
    hit_count: int


class ObserverStateStore:
    """Read helper for Observer meta_forest state nodes."""

    def __init__(self, meta_forest: TieredForest) -> None:
        self.meta_forest = meta_forest

    def node_state(self, node_id: str) -> NodeStateSnapshot | None:
        s = self.meta_forest.get(f"state:{node_id}")
        if s is None:
            return None
        return NodeStateSnapshot(
            weight=float(s.mu[0]),
            score=float(s.mu[1]),
            miss_count=int(s.mu[2]),
            hit_count=int(s.mu[3]),
        )

    def miss_count(self, node_id: str) -> int:
        snapshot = self.node_state(node_id)
        return snapshot.miss_count if snapshot is not None else 0

    def weights_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for sn in self.meta_forest.active_nodes():
            if sn.id.startswith("state:"):
                out[sn.id[len("state:"):]] = float(sn.mu[0])
        return out
