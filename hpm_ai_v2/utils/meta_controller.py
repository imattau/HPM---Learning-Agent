"""
MetaStrategyController and SolveRecord — HPM Level 5 meta-cognition.

Ported from experiment_meta_strategy_controller.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hfn.hfn import HFN


@dataclass
class SolveRecord:
    """Captures outcome of one solve attempt."""
    task_id: str
    goal_type: str       # "scalar" | "map" | "filter"
    n_macros: int        # macros registered at solve time
    strategy: str        # "exact" | "decompose" | "imagine" | "bfs"
    depth: int
    oracle_calls: int
    success: bool
    wall_ms: float


class MetaStrategyController:
    """
    Learns a strategy selection policy from SolveRecord history.

    Context key: (goal_type, n_macros_bucket)
      n_macros_bucket: 0 = none, 1 = one, 2 = two or more

    After each solve, updates success counts for (context, strategy).
    rank_strategies() returns strategies sorted by success rate desc,
    tie-breaking by mean oracle_calls asc.
    """

    DEFAULT_ORDER: List[str] = [
        "exact",
        "decompose",
        "imagine",
        "bfs",
        "analogy",
        "social",
        "recombine",
        "compose",
    ]

    def __init__(self) -> None:
        # (context_key, strategy) -> [successes, attempts, total_oracle_calls]
        self._stats: Dict[Tuple, Dict[str, List[int]]] = defaultdict(
            lambda: {s: [0, 0, 0] for s in self.DEFAULT_ORDER}
        )
        self.history: List[SolveRecord] = []

    def _bucket(self, n_macros: int) -> int:
        if n_macros == 0:
            return 0
        if n_macros == 1:
            return 1
        return 2

    def _context_key(self, goal_type: str, n_macros: int) -> Tuple:
        return (goal_type, self._bucket(n_macros))

    def record(self, rec: SolveRecord, pattern_used: Optional[HFN] = None) -> None:
        """Update strategy success rates for this context."""
        self.history.append(rec)
        key = self._context_key(rec.goal_type, rec.n_macros)
        if rec.strategy not in self._stats[key]:
            self._stats[key][rec.strategy] = [0, 0, 0]
        entry = self._stats[key][rec.strategy]
        entry[1] += 1                    # attempts
        entry[2] += rec.oracle_calls     # total oracle calls
        if rec.success:
            entry[0] += 1               # successes

    def rank_strategies(self, goal_type: str, n_macros: int) -> List[str]:
        """Return strategies sorted by historical success rate (desc), then
        oracle calls (asc). Falls back to DEFAULT_ORDER if no history."""
        key = self._context_key(goal_type, n_macros)
        stats = self._stats.get(key)
        if stats is None:
            return list(self.DEFAULT_ORDER)
        has_history = any(v[1] > 0 for v in stats.values())
        if not has_history:
            return list(self.DEFAULT_ORDER)

        def sort_key(strategy: str) -> Tuple[float, float]:
            successes, attempts, oracle_total = stats.get(strategy, [0, 0, 0])
            rate = successes / attempts if attempts > 0 else 0.0
            avg_calls = oracle_total / attempts if attempts > 0 else float('inf')
            return (-rate, avg_calls)

        return sorted(self.DEFAULT_ORDER, key=sort_key)

    def meta_patterns(self) -> List[str]:
        """Return a list of discovered meta-patterns (context + best strategy)."""
        patterns = []
        for key, strat_dict in self._stats.items():
            goal_type, bucket = key
            ranked = self.rank_strategies(goal_type, bucket)
            best = ranked[0]
            entry = strat_dict.get(best, [0, 0, 0])
            if entry[1] > 0:
                rate = entry[0] / entry[1]
                patterns.append(
                    f"context=({goal_type},macros_bucket={bucket}) "
                    f"-> best_strategy={best} (rate={rate:.2f}, "
                    f"attempts={entry[1]})"
                )
        return patterns
