"""
HFNMetaStrategyController — stores meta-strategy performance as HFN nodes.
Empowers L5 to be fractal-consistent by representing search strategies in persistent nodes.
"""
from __future__ import annotations
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from hfn.tiered_forest import TieredForest
from hfn.hfn import HFN
from hfn.recombination import Recombination

if TYPE_CHECKING:
    from hpm_ai_v2.utils.meta_controller import SolveRecord

class HFNMetaStrategyController:
    """
    A meta-strategy controller that ranks search strategies based on HFN-stored stats.
    Uses solve record nodes and aggregate meta-nodes for fractal consistency.
    """
    DEFAULT_ORDER = [
        "exact", "decompose", "imagine", "bfs", "analogy",
        "social", "recombine", "compose"
    ]

    def __init__(self, cold_dir: Optional[str | Path] = None, max_history: int = 100):
        self.meta_forest = TieredForest(
            D=4,  # [successes, attempts, total_oracle_calls, last_timestamp]
            cold_dir=cold_dir or Path("data/knowledge_base/meta_controller"),
            forest_id="meta"
        )
        self.recombination = Recombination()
        self.history: List["SolveRecord"] = []
        self.max_history = max_history
        # In-memory index for quick access to solve records per (context, strategy)
        self._solve_records: Dict[Tuple[str, str], List[HFN]] = {}

    def _bucket(self, n_macros: int) -> int:
        """Bucket macro count: 0 -> 0, 1 -> 1, >=2 -> 2."""
        return 0 if n_macros == 0 else 1 if n_macros == 1 else 2

    def record(self, rec: "SolveRecord", pattern_used: Optional[HFN] = None) -> None:
        """Record the result of a strategy attempt using fractal nodes."""
        self.history.append(rec)
        context = f"{rec.goal_type}:{self._bucket(rec.n_macros)}"
        strategy = rec.strategy
        
        # 1. Create Solve Record Node
        record_node = self._create_solve_record(context, strategy, pattern_used, rec.success, rec.oracle_calls, rec.depth)
        
        # 2. Update Aggregate Meta-Node
        self._update_meta_node(context, strategy, record_node)

    def _create_solve_record(self, context: str, strategy: str, pattern_used: Optional[HFN], 
                            success: bool, oracle_calls: int, depth: int) -> HFN:
        """Create a node representing a single solve attempt."""
        node_id = f"solve:{context}:{strategy}:{int(time.time()*1000)}"
        # mu = [success (1/0), oracle_calls, depth, timestamp]
        mu = np.array([1.0 if success else 0.0, float(oracle_calls), float(depth), time.time()])
        node = HFN(mu=mu, sigma=np.ones(4), id=node_id, use_diag=True)
        if pattern_used:
            node.inputs = [pattern_used]
        node.relation_type = "solve_record"
        self.meta_forest.register(node)
        return node

    def _update_meta_node(self, context: str, strategy: str, new_record: HFN) -> None:
        """Update the aggregate meta-node for a given context and strategy."""
        key = (context, strategy)
        if key not in self._solve_records:
            self._solve_records[key] = []
        
        solve_nodes = self._solve_records[key]
        solve_nodes.append(new_record)
        if len(solve_nodes) > self.max_history:
            # We don't deregister old solve records from forest, just stop aggregating them
            self._solve_records[key] = solve_nodes[-self.max_history:]
            solve_nodes = self._solve_records[key]

        agg_id = f"meta:{context}:{strategy}"
        existing = self.meta_forest.get(agg_id)
        
        # Compute aggregated mu: [avg_success_rate, avg_oracle_calls, count, last_timestamp]
        successes = sum(1 for n in solve_nodes if n.mu[0] > 0.5)
        total_calls = sum(n.mu[1] for n in solve_nodes)
        new_mu = np.array([
            successes / len(solve_nodes),
            total_calls / len(solve_nodes),
            float(len(solve_nodes)),
            time.time()
        ])
        
        if existing:
            self.meta_forest.deregister(existing.id)
            
        new_agg = self.recombination.aggregate(
            solve_nodes, lambda mus: new_mu, agg_id, "meta_pattern"
        )
        self.meta_forest.register(new_agg)

    def rank_strategies(self, goal_type: str, n_macros: int) -> List[str]:
        """Rank strategies based on aggregate meta-node stats."""
        bucket = self._bucket(n_macros)
        prefix = f"meta:{goal_type}:{bucket}:"
        
        candidates: List[Tuple[str, float, float]] = []
        
        # Each meta-node id is meta:{goal_type}:{bucket}:{strategy}
        for node in self.meta_forest.active_nodes():
            if node.id.startswith(prefix) and node.relation_type == "meta_pattern":
                strategy = node.id.split(":")[-1]
                rate, avg_calls, count, _ = node.mu
                if count > 0:
                    candidates.append((strategy, rate, avg_calls))
        
        if not candidates:
            return list(self.DEFAULT_ORDER)
            
        # Sort by higher success rate, then lower average oracle calls
        candidates.sort(key=lambda x: (-x[1], x[2]))
        
        ranked = [c[0] for c in candidates]
        for s in self.DEFAULT_ORDER:
            if s not in ranked:
                ranked.append(s)
                
        return ranked

    def save_state(self) -> None:
        """Persist meta-stats to cold storage."""
        self.meta_forest.save_to_cold()
