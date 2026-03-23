"""
hpm/monitor/structural_law.py — Structural Law Monitor ("The Librarian")

Observes the shared PatternStore population and computes Field Quality Metrics.
Integrated into MultiAgentOrchestrator as an optional monitor= parameter.

Light metrics: computed every step (inexpensive).
Heavy metrics: computed every T_monitor steps (O(n²) pairwise operations).
"""

import json
import math
from typing import Any

import numpy as np

from hpm.dynamics.meta_pattern_rule import sym_kl_normalised


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _taboo_overlap(agents) -> float:
    """
    Fraction of negative pattern UUIDs present in ALL agents vs any agent (Jaccard).
    Returns 0.0 if no agents have negative patterns.
    """
    neg_id_sets = [
        {p.id for p, _ in agent.store.query_negative(agent.agent_id)}
        for agent in agents
        if hasattr(agent.store, 'query_negative')
    ]
    # Filter out agents with empty negative stores for union/intersection purposes
    non_empty = [s for s in neg_id_sets if s]
    if not non_empty:
        return 0.0
    union = non_empty[0].union(*non_empty[1:])
    if not union:
        return 0.0
    intersection = non_empty[0].intersection(*non_empty[1:])
    return len(intersection) / len(union)


class StructuralLawMonitor:
    """
    Computes population-level Field Quality Metrics from a shared PatternStore.

    Args:
        store:              Shared PatternStore (SQLiteStore recommended).
        field:              Optional PatternField (reserved; unused in v1).
        T_monitor:          Cadence for heavy metrics and console/log output.
        log_path:           NDJSON log path; None disables logging.
        conflict_threshold: Threshold above which pattern pairs count as conflicting.
    """

    def __init__(
        self,
        store,
        field=None,
        T_monitor: int = 50,
        log_path: str | None = None,
        conflict_threshold: float = 0.5,
        verbose: bool = True,
    ):
        self._store = store
        self._field = field
        self._T_monitor = T_monitor
        self._log_path = log_path
        self._conflict_threshold = conflict_threshold
        self._verbose = verbose
        self._t = 0

    def step(self, step_t: int, agents: list, total_conflict: float) -> dict:
        """
        Called by MultiAgentOrchestrator after each step.

        Returns field_quality dict:
          - Light metrics always present.
          - Heavy metrics (diversity, redundancy) present every T_monitor steps; else None.
        """
        self._t += 1

        all_records = self._store.query_all()  # list of (pattern, weight, agent_id)
        patterns = [p for p, _, _ in all_records]
        weights = [w for _, w, _ in all_records]

        light = self._compute_light(patterns, weights, total_conflict)

        negative_count = sum(
            len(agent.store.query_negative(agent.agent_id))
            for agent in agents
            if hasattr(agent.store, 'query_negative')
        )
        taboo_ov = _taboo_overlap(agents)
        light["negative_count"] = negative_count
        light["taboo_overlap"] = taboo_ov

        heavy_diversity = None
        heavy_redundancy = None

        if self._t % self._T_monitor == 0:
            heavy_diversity, heavy_redundancy = self._compute_heavy(patterns, weights)
            if self._verbose:
                self._print_table(step_t, light, heavy_diversity, heavy_redundancy)
            if self._log_path is not None:
                self._log_json(step_t, light, heavy_diversity, heavy_redundancy)

        return {
            **light,
            "diversity": heavy_diversity,
            "redundancy": heavy_redundancy,
            "negative_count": light.get("negative_count", 0),
            "taboo_overlap": light.get("taboo_overlap", 0.0),
        }

    # ------------------------------------------------------------------
    # Light metrics
    # ------------------------------------------------------------------

    def _compute_light(self, patterns, weights, total_conflict: float) -> dict:
        level_dist = {lvl: 0 for lvl in range(1, 6)}
        for p in patterns:
            lvl = getattr(p, "level", 1)
            level_dist[min(max(lvl, 1), 5)] += 1

        l4plus = [(p, w) for p, w in zip(patterns, weights) if getattr(p, "level", 1) >= 4]
        l4plus_count = len(l4plus)
        l4plus_mean_weight = (
            float(np.mean([w for _, w in l4plus])) if l4plus else 0.0
        )

        stability_mean = (
            float(np.mean([_sigmoid(getattr(p, "level", 1) / 5.0) for p in patterns]))
            if patterns else 0.0
        )

        return {
            "pattern_count": len(patterns),
            "level_distribution": level_dist,
            "level4plus_count": l4plus_count,
            "level4plus_mean_weight": l4plus_mean_weight,
            "conflict": float(total_conflict),
            "stability_mean": stability_mean,
        }

    # ------------------------------------------------------------------
    # Heavy metrics
    # ------------------------------------------------------------------

    def _compute_heavy(self, patterns, weights) -> tuple[float, float]:
        # Diversity: entropy of weight distribution
        total_w = sum(weights)
        if total_w > 0:
            norm_weights = [w / total_w for w in weights]
        else:
            norm_weights = [1.0 / len(weights)] * len(weights) if weights else []
        diversity = max(0.0, -sum(w * math.log(w + 1e-9) for w in norm_weights)) if norm_weights else 0.0

        # Redundancy: 1.0 - mean pairwise sym_kl_normalised for Level 4+ patterns
        # High similarity (low KL) -> High redundancy
        l4plus_patterns = [p for p in patterns if getattr(p, "level", 1) >= 4]
        if len(l4plus_patterns) < 2:
            redundancy = 0.0
        else:
            sims = []
            for i in range(len(l4plus_patterns)):
                for j in range(i + 1, len(l4plus_patterns)):
                    sims.append(sym_kl_normalised(l4plus_patterns[i], l4plus_patterns[j]))
            redundancy = float(1.0 - np.mean(sims)) if sims else 0.0

        return diversity, redundancy

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _print_table(self, step_t, light, diversity, redundancy):
        title = f"Field Quality Report (step {step_t})"
        cols = ["Patterns", "L4+", "L4+ Weight", "Diversity", "Redundancy",
                "Conflict", "Stable", "NegCount", "TabooOvlp"]
        row = {
            "Patterns": str(light["pattern_count"]),
            "L4+": str(light["level4plus_count"]),
            "L4+ Weight": f"{light['level4plus_mean_weight']:.2f}",
            "Diversity": f"{diversity:.2f}" if diversity is not None else "—",
            "Redundancy": f"{redundancy:.2f}" if redundancy is not None else "—",
            "Conflict": f"{light['conflict']:.2f}",
            "Stable": f"{light['stability_mean']:.2f}",
            "NegCount": str(light.get("negative_count", 0)),
            "TabooOvlp": f"{light.get('taboo_overlap', 0.0):.2f}",
        }
        col_widths = {c: max(len(c), len(row[c])) for c in cols}
        sep = "   "
        header = sep.join(c.ljust(col_widths[c]) for c in cols)
        total_width = max(len(title), len(header))
        print()
        print(title)
        print("─" * total_width)
        print(header)
        print("─" * total_width)
        print(sep.join(row[c].ljust(col_widths[c]) for c in cols))
        print()

    def _log_json(self, step_t, light, diversity, redundancy):
        entry = {
            "step": step_t,
            **light,
            "level_distribution": {str(k): v for k, v in light["level_distribution"].items()},
            "diversity": diversity,
            "redundancy": redundancy,
            "negative_count": light.get("negative_count", 0),
            "taboo_overlap": light.get("taboo_overlap", 0.0),
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
