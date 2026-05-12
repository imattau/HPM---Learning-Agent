from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner
from hpm_ai_v6.hpm_model.storage.pattern_pager import PatternPager

class BaseHPMAgent(ABC):
    """
    Abstract Base Class for HPM Agents.
    Defines the standard interface for perceiving and learning from pattern sequences.
    """
    def __init__(self, 
                 patterns: List[Cell],
                 learning_rate: float = 0.2,
                 conflict_scale: float = 0.05,
                 forget_decay: float = 1.0,
                 beta_e: float = 1.0,
                 beta_a: float = 0.5,
                 beta_s: float = 0.5,
                 pattern_cache_dir: Optional[str] = None,
                 max_active_patterns: int = 0):
        self.patterns = patterns
        self.pattern_cache_dir = pattern_cache_dir
        self.max_active_patterns = max_active_patterns
        self.pattern_pager = None
        if self.pattern_cache_dir and self.max_active_patterns > 0:
            agent_name = self.__class__.__name__.lower()
            self.pattern_pager = PatternPager(
                cache_dir=self.pattern_cache_dir,
                agent_name=agent_name,
                max_active_patterns=self.max_active_patterns,
            )
        self.meta_rule = MetaPatternRule(
            patterns=self.patterns,
            learning_rate=learning_rate,
            conflict_scale=conflict_scale,
            forget_decay=forget_decay
        )
        self.learner = HPMLearner(
            meta_rule=self.meta_rule,
            beta_e=beta_e,
            beta_a=beta_a,
            beta_s=beta_s
        )

    @abstractmethod
    def perceive(self, observation_seq: List[Cell], population: List[Cell], context: Dict[str, Any]):
        """Agent processes a sequence of observations and updates its internal state."""
        pass

    def get_best_pattern(self) -> Cell:
        return self.meta_rule.get_best_pattern()

    def get_weights_tensor(self) -> torch.Tensor:
        return self.meta_rule.get_weights_tensor().clone()

    def get_weights(self) -> np.ndarray:
        return self.meta_rule.weights.copy()

    def get_weights_dict(self) -> Dict[str, float]:
        return self.meta_rule.get_weights_dict()

    def _paging_lookup(self) -> Dict[str, Cell]:
        return {}

    def warm_start(self, patterns: List[Cell], weights: List[float]) -> None:
        """Seed the agent with a list of patterns and their weights."""
        if not patterns:
            return

        weight_map = {p.name: w for p, w in zip(patterns, weights)}
        existing_names = {p.name for p in self.patterns}
        
        added = False
        for p in patterns:
            if p.name not in existing_names:
                self.patterns.append(p)
                existing_names.add(p.name)
                added = True
        
        if not added:
            return

        if hasattr(self, "_refresh_learner"):
            self._refresh_learner()
        
        # Update weights in the meta_rule if it exists and was potentially refreshed
        if hasattr(self, "meta_rule"):
            current_weights = self.get_weights()
            for i, p in enumerate(self.patterns):
                if p.name in weight_map:
                    current_weights[i] = weight_map[p.name]
            
            # Normalize
            total = np.sum(current_weights)
            if total > 0:
                self.meta_rule.weights = current_weights / total

    def restore_pattern_from_archive(
        self,
        name: str,
        query_embedding: Optional[Any] = None,
        min_similarity: float = 0.7,
    ) -> Optional[Cell]:
        if self.pattern_pager is None:
            return None

        for pattern in self.patterns:
            if pattern.name == name:
                return pattern

        restored = self.pattern_pager.load(name, self._paging_lookup())
        if restored is None and query_embedding is not None:
            restored = self.pattern_pager.load_nearest(
                query_embedding,
                self._paging_lookup(),
                min_similarity=min_similarity,
                exclude_names=[name],
            )
        if restored is None:
            return None

        for pattern in self.patterns:
            if pattern.name == restored.name:
                return pattern

        self.patterns.append(restored)
        return restored

    def hydrate_patterns_from_archive(
        self,
        limit: Optional[int] = None,
        min_weight: float = 0.0,
    ) -> int:
        if self.pattern_pager is None:
            return 0

        payloads = self.pattern_pager.iter_index_payloads()
        if not payloads:
            return 0

        payloads = sorted(payloads, key=lambda payload: float(payload.get("weight", 0.0)), reverse=True)
        loaded = 0
        existing = {pattern.name for pattern in self.patterns}
        lookup = self._paging_lookup()

        for payload in payloads:
            if limit is not None and loaded >= limit:
                break
            try:
                weight = float(payload.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            if weight < min_weight:
                continue

            name = str(payload.get("name", ""))
            if not name or name in existing:
                continue

            restored = self.pattern_pager.load_from_payload(payload, lookup)
            if restored.name in existing:
                continue

            self.patterns.append(restored)
            existing.add(restored.name)
            loaded += 1

        if loaded and hasattr(self, "_refresh_learner"):
            self._refresh_learner()
        return loaded

    def maybe_page_patterns(self) -> int:
        if self.pattern_pager is None or self.max_active_patterns <= 0:
            return 0
        if len(self.patterns) <= self.max_active_patterns:
            return 0

        weights = self.get_weights()
        evict_indices = self.pattern_pager.select_evictions(self.patterns, weights)
        if not evict_indices:
            return 0

        evicted = [self.patterns[idx] for idx in evict_indices]
        for pattern in evicted:
            self.pattern_pager.save(pattern)

        evict_names = {pattern.name for pattern in evicted}
        self.patterns = [pattern for pattern in self.patterns if pattern.name not in evict_names]

        if hasattr(self, "_refresh_learner"):
            self._refresh_learner()

        return len(evicted)

    def flush_pager(self) -> None:
        if self.pattern_pager is not None:
            self.pattern_pager.flush()

    def close_pager(self) -> None:
        if self.pattern_pager is not None:
            self.pattern_pager.flush()
            self.pattern_pager.close()
