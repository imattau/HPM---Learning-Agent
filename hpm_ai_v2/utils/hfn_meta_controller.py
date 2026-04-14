"""
HFNMetaStrategyController — stores meta-strategy performance as HFN nodes.
Empowers L5 to be fractal-consistent by representing search strategies in persistent nodes.
"""
from __future__ import annotations
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING
from hfn.tiered_forest import TieredForest
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hpm_ai_v2.utils.meta_controller import SolveRecord

class HFNMetaStrategyController:
    """
    A meta-strategy controller that ranks search strategies based on HFN-stored stats.
    Each combination of (goal_type, macros_bucket, strategy) is a persistent node.
    """
    DEFAULT_ORDER = [
        "exact", "decompose", "imagine", "bfs", "analogy",
        "social", "recombine", "compose"
    ]

    def __init__(self, cold_dir: Optional[str | Path] = None):
        self.meta_forest = TieredForest(
            D=4,  # [successes, attempts, total_oracle_calls, last_timestamp]
            cold_dir=cold_dir or Path("data/knowledge_base/meta_controller"),
            forest_id="meta"
        )
        self.history: List["SolveRecord"] = []

    def _node_id(self, goal_type: str, bucket: int, strategy: str) -> str:
        return f"meta:{goal_type}:{bucket}:{strategy}"

    def _bucket(self, n_macros: int) -> int:
        """Bucket macro count: 0 -> 0, 1 -> 1, >=2 -> 2."""
        return 0 if n_macros == 0 else 1 if n_macros == 1 else 2

    def record(self, rec: "SolveRecord") -> None:
        """Record the result of a strategy attempt."""
        self.history.append(rec)
        nid = self._node_id(rec.goal_type, self._bucket(rec.n_macros), rec.strategy)
        
        node = self.meta_forest.get(nid)
        if node is None:
            # Initialize metrics
            node = HFN(
                mu=np.zeros(4),
                sigma=np.ones(4),
                id=nid,
                use_diag=True
            )
            self.meta_forest.register(node)
        
        # Update metrics in mu vector
        # mu = [successes, attempts, total_oracle_calls, last_timestamp]
        node.mu[0] += 1.0 if rec.success else 0.0
        node.mu[1] += 1.0
        node.mu[2] += float(rec.oracle_calls)
        node.mu[3] = time.time()

    def rank_strategies(self, goal_type: str, n_macros: int) -> List[str]:
        """Rank strategies based on HFN stats."""
        bucket = self._bucket(n_macros)
        prefix = f"meta:{goal_type}:{bucket}:"
        
        candidates: List[Tuple[str, float, float]] = []
        
        # Retrieve active nodes from the meta forest
        for node in self.meta_forest.active_nodes():
            if node.id.startswith(prefix):
                # node_id = meta:{goal_type}:{bucket}:{strategy}
                strategy = node.id.split(":")[-1]
                
                successes, attempts, total_calls, _ = node.mu
                if attempts > 0:
                    rate = successes / attempts
                    avg_calls = total_calls / attempts
                    candidates.append((strategy, rate, avg_calls))
        
        if not candidates:
            return list(self.DEFAULT_ORDER)
            
        # Sort by higher success rate, then lower average oracle calls
        candidates.sort(key=lambda x: (-x[1], x[2]))
        
        # Include any default strategies not yet in the forest
        ranked = [c[0] for c in candidates]
        for s in self.DEFAULT_ORDER:
            if s not in ranked:
                ranked.append(s)
                
        return ranked

    def save_state(self) -> None:
        """Persist meta-stats to cold storage."""
        self.meta_forest.save_to_cold()
