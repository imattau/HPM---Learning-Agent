"""
L3RelationalMixin — HPM Level 3: relational rules and meta-schemas.

Provides:
- discover_meta_schema(solved_paths): extract shared structure across solutions
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.utils.state import S_DIM, DIM


class L3RelationalMixin:
    """
    Mixin adding HPM L3 relational/meta-schema discovery.

    Must be used with BaseHFNAgent + L2MacroMixin.
    """

    def discover_meta_schema(
        self,
        solved_paths: List[List[HFN]],
        min_support: int = 2,
    ) -> Optional[HFN]:
        """
        Extract a meta-schema by finding the most common node sub-sequence
        across a collection of solved paths.

        Args:
            solved_paths: list of node paths that successfully solved tasks
            min_support: minimum number of paths a sub-sequence must appear in

        Returns:
            A new HFN node with relation_type='meta_schema' if a schema is found,
            else None.
        """
        if not solved_paths or len(solved_paths) < min_support:
            return None

        # Count how often each (node_id, ...) bigram appears across paths
        bigram_counts: Dict[Tuple[str, ...], int] = {}
        bigram_examples: Dict[Tuple[str, ...], List[List[HFN]]] = {}

        for path in solved_paths:
            ids = [n.id for n in path]
            seen_in_path: set = set()
            for i in range(len(ids)):
                for j in range(i + 1, min(i + 4, len(ids) + 1)):
                    key = tuple(ids[i:j])
                    if key not in seen_in_path:
                        seen_in_path.add(key)
                        bigram_counts[key] = bigram_counts.get(key, 0) + 1
                        bigram_examples.setdefault(key, []).append(path[i:j])

        # Find most-supported sub-sequence (prefer longer)
        best_key: Optional[Tuple[str, ...]] = None
        best_score = 0
        for key, count in bigram_counts.items():
            if count >= min_support:
                score = count * len(key)  # weight by length
                if score > best_score:
                    best_score = score
                    best_key = key

        if best_key is None:
            return None

        # Build meta-schema node from the representative example
        representative = bigram_examples[best_key][0]
        composed = self._compose_sequence(representative)
        if composed is None:
            return None

        schema_id = f"meta_schema_{'_'.join(best_key[:3])}"
        composed.id = schema_id
        composed.relation_type = "meta_schema"

        self.patterns[schema_id] = composed
        if schema_id not in self.forest:
            self.observer.register(composed, protected=False, initial_weight=0.2)

        return composed
