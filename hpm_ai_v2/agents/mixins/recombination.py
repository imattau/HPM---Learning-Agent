"""
RecombinationMixin — cross-domain pattern recombination (analogical transfer).

Provides:
- recombine_patterns(p1, p2): generate a novel pattern by blending two
- insight_score(node, inputs, outputs): score a node on unseen task
- _try_recombine(inputs, outputs): solve by recombining existing patterns
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.utils.state import S_DIM, DIM


class RecombinationMixin:
    """
    Mixin adding analogical recombination of stored patterns.

    Must be used with BaseHFNAgent (provides self.patterns, self.renderer,
    self.executor, self._check_outputs, self._compose_sequence, self.m_dim).
    """

    def recombine_patterns(
        self,
        name_a: str,
        name_b: str,
        blend_weight: float = 0.5,
        new_name: Optional[str] = None,
    ) -> Optional[HFN]:
        """
        Generate a novel pattern by interpolating the mu vectors of two patterns.

        The resulting node is registered as a new pattern.

        Args:
            name_a: key into self.patterns for the first parent
            name_b: key into self.patterns for the second parent
            blend_weight: weight for pattern A (1-blend_weight for B)
            new_name: optional name for the new pattern (default: 'recomb_A_B')

        Returns:
            The new HFN node, or None if either pattern is missing.
        """
        node_a = self.patterns.get(name_a)
        node_b = self.patterns.get(name_b)
        if node_a is None or node_b is None:
            return None

        mu_new = blend_weight * node_a.mu + (1.0 - blend_weight) * node_b.mu
        sigma_new = np.ones(self.m_dim)

        result_name = new_name or f"recomb_{name_a}_{name_b}"
        node_new = HFN(
            mu=mu_new,
            sigma=sigma_new,
            id=f"recomb_{name_a}_{name_b}",
            inputs=[node_a, node_b],
            relation_type="macro",
            use_diag=True,
        )
        self.patterns[result_name] = node_new
        if node_new.id not in self.forest:
            self.observer.register(node_new, protected=False, initial_weight=0.2)
        return node_new

    def insight_score(
        self,
        node: HFN,
        inputs: List[Any],
        outputs: List[Any],
    ) -> float:
        """
        Score a node on a task (0.0 = fail, 1.0 = perfect match).

        Used to rank recombined patterns before committing to a solve attempt.
        """
        code = self.renderer.render(node)
        if not code:
            return 0.0
        results, errors = self.executor.run_batch(code, inputs)
        if self._check_outputs(results, outputs):
            return 1.0
        # Partial credit: fraction of outputs correct
        if len(results) == 0 or len(outputs) == 0:
            return 0.0
        n_correct = sum(1 for r, e in zip(results, outputs) if r == e)
        return n_correct / len(outputs)

    def _try_recombine(
        self,
        inputs: List[Any],
        outputs: List[Any],
        blend_steps: int = 3,
    ) -> Optional[List[HFN]]:
        """
        Strategy: generate recombinations of all pattern pairs, rank by
        insight_score, return the best if it solves the task.
        """
        pattern_names = list(self.patterns.keys())
        if len(pattern_names) < 2:
            return None

        candidates: List[Tuple[float, HFN]] = []
        weights = [0.3, 0.5, 0.7][:blend_steps]

        for i in range(len(pattern_names)):
            for j in range(i + 1, len(pattern_names)):
                for w in weights:
                    recomb = self.recombine_patterns(
                        pattern_names[i],
                        pattern_names[j],
                        blend_weight=w,
                    )
                    if recomb is not None:
                        score = self.insight_score(recomb, inputs, outputs)
                        candidates.append((score, recomb))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_node = candidates[0]
        if best_score >= 1.0:
            return [best_node]
        return None
