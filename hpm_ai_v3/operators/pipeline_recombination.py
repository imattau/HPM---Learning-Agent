"""
pipeline_recombination.py - Operator that creates composite patterns from frequently co-occurring tools.
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from ..pattern import HPMPattern
from ..tools.base import ToolPattern
from ..agents.base import AgentPattern
from ..tools.composite import CompositeToolPattern
from ..agents.composite import CompositeAgentPattern
from ..population import PatternPopulation


class PipelineRecombinationOperator:
    """
    Observes sequences of pattern invocations (tools or agents) and creates 
    Composite patterns from frequently co-occurring pairs or chains.
    """
    def __init__(self, 
                 co_occurrence_threshold: float = 0.1,
                 min_weight: float = 0.05,
                 max_pipeline_length: int = 3):
        """
        Args:
            co_occurrence_threshold: Minimum frequency ratio to consider a pair.
            min_weight: Minimum pattern weight to be considered for recombination.
            max_pipeline_length: Maximum number of patterns in a composite.
        """
        self.co_occurrence_threshold = co_occurrence_threshold
        self.min_weight = min_weight
        self.max_pipeline_length = max_pipeline_length
        
        # Co-occurrence counts: (name1, name2) -> count
        self.pair_counts = defaultdict(int)
        self.total_sequences = 0
        
        # Pattern usage counts for normalization
        self.usage_counts = defaultdict(int)
        
    def record_sequence(self, name_sequence: List[str]):
        """
        Record a sequence of pattern names as they were invoked.
        """
        if len(name_sequence) < 2:
            return
        
        self.total_sequences += 1
        for name in name_sequence:
            self.usage_counts[name] += 1
        
        # Record adjacent pairs
        for i in range(len(name_sequence) - 1):
            pair = (name_sequence[i], name_sequence[i+1])
            self.pair_counts[pair] += 1
    
    def get_pair_probability(self, name1: str, name2: str) -> float:
        """
        Probability of name2 following name1 given name1 occurred.
        """
        if self.usage_counts[name1] == 0:
            return 0.0
        return self.pair_counts.get((name1, name2), 0) / self.usage_counts[name1]
    
    def find_frequent_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Return pairs with conditional probability above threshold.
        """
        frequent = []
        for (n1, n2), count in self.pair_counts.items():
            prob = self.get_pair_probability(n1, n2)
            if prob >= self.co_occurrence_threshold:
                frequent.append((n1, n2, prob))
        return sorted(frequent, key=lambda x: x[2], reverse=True)
    
    def should_recombine(self, population: PatternPopulation) -> Optional[HPMPattern]:
        """
        Check population for high-weight, frequently co-occurring pattern pairs
        and create a composite pattern if found.
        """
        # Collect all discoverable patterns in population with sufficient weight
        discoverable_patterns: Dict[str, HPMPattern] = {}
        for p in population.patterns:
            if p.weight < self.min_weight:
                continue
                
            if isinstance(p, ToolPattern):
                discoverable_patterns[p.tool_name] = p
            elif isinstance(p, AgentPattern):
                discoverable_patterns[p.agent_name] = p
        
        if len(discoverable_patterns) < 2:
            return None
        
        # Find frequent pairs where both are in population
        frequent_pairs = self.find_frequent_pairs()
        for n1, n2, prob in frequent_pairs:
            if n1 in discoverable_patterns and n2 in discoverable_patterns:
                pat1 = discoverable_patterns[n1]
                pat2 = discoverable_patterns[n2]
                
                # Case 1: Both are tools -> CompositeToolPattern
                if isinstance(pat1, ToolPattern) and isinstance(pat2, ToolPattern):
                    if self._find_existing_tool_composite(population, [n1, n2]):
                        continue
                    composite = CompositeToolPattern([pat1, pat2])
                
                # Case 2: Both are agents -> CompositeAgentPattern
                elif isinstance(pat1, AgentPattern) and isinstance(pat2, AgentPattern):
                    if self._find_existing_agent_composite(population, [n1, n2]):
                        continue
                    composite = CompositeAgentPattern([pat1, pat2])
                
                # Case 3: Mixed (wrap tool as agent-like if needed, or vice-versa)
                # For now, we only recombine like-with-like for simplicity
                else:
                    continue
                
                composite.weight = 0.01
                composite.insight_boost = prob * 0.5
                return composite
        
        return None
    
    def _find_existing_tool_composite(self, population: PatternPopulation, tool_names: List[str]) -> Optional[CompositeToolPattern]:
        """Check if a composite with exact same tool sequence already exists."""
        for p in population.patterns:
            if isinstance(p, CompositeToolPattern):
                seq = [pt.tool_name for pt in p.patterns]
                if seq == tool_names:
                    return p
        return None
        
    def _find_existing_agent_composite(self, population: PatternPopulation, agent_names: List[str]) -> Optional[CompositeAgentPattern]:
        """Check if a composite with exact same agent sequence already exists."""
        for p in population.patterns:
            if isinstance(p, CompositeAgentPattern):
                seq = [pa.agent_name for pa in p.patterns]
                if seq == agent_names:
                    return p
        return None
