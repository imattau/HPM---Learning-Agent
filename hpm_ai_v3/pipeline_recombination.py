"""
pipeline_recombination.py - Operator that creates composite patterns from frequently co-occurring tools.
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from .pattern import HPMPattern
from .tool_pattern import ToolPattern
from .composite_tool_pattern import CompositeToolPattern
from .population import PatternPopulation


class PipelineRecombinationOperator:
    """
    Observes sequences of tool invocations and creates CompositeToolPatterns
    from frequently co-occurring pairs or chains.
    """
    def __init__(self, 
                 co_occurrence_threshold: float = 0.1,
                 min_weight: float = 0.05,
                 max_pipeline_length: int = 3):
        """
        Args:
            co_occurrence_threshold: Minimum frequency ratio to consider a pair.
            min_weight: Minimum pattern weight to be considered for recombination.
            max_pipeline_length: Maximum number of tools in a composite pattern.
        """
        self.co_occurrence_threshold = co_occurrence_threshold
        self.min_weight = min_weight
        self.max_pipeline_length = max_pipeline_length
        
        # Co-occurrence counts: (tool1_name, tool2_name) -> count
        self.pair_counts = defaultdict(int)
        self.total_sequences = 0
        
        # Tool usage counts for normalization
        self.tool_counts = defaultdict(int)
        
    def record_sequence(self, tool_sequence: List[str]):
        """
        Record a sequence of tool names as they were invoked.
        """
        if len(tool_sequence) < 2:
            return
        
        self.total_sequences += 1
        for name in tool_sequence:
            self.tool_counts[name] += 1
        
        # Record adjacent pairs
        for i in range(len(tool_sequence) - 1):
            pair = (tool_sequence[i], tool_sequence[i+1])
            self.pair_counts[pair] += 1
    
    def get_pair_probability(self, tool1: str, tool2: str) -> float:
        """
        Probability of tool2 following tool1 given tool1 occurred.
        """
        if self.tool_counts[tool1] == 0:
            return 0.0
        return self.pair_counts.get((tool1, tool2), 0) / self.tool_counts[tool1]
    
    def find_frequent_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Return pairs with conditional probability above threshold.
        """
        frequent = []
        for (t1, t2), count in self.pair_counts.items():
            prob = self.get_pair_probability(t1, t2)
            if prob >= self.co_occurrence_threshold:
                frequent.append((t1, t2, prob))
        return sorted(frequent, key=lambda x: x[2], reverse=True)
    
    def should_recombine(self, population: PatternPopulation) -> Optional[CompositeToolPattern]:
        """
        Check population for high-weight, frequently co-occurring tool pairs
        and create a composite pattern if found.
        """
        # Find all ToolPatterns in population with sufficient weight
        tool_patterns: Dict[str, ToolPattern] = {}
        for p in population.patterns:
            if isinstance(p, ToolPattern) and p.weight >= self.min_weight:
                tool_patterns[p.tool_name] = p
        
        if len(tool_patterns) < 2:
            return None
        
        # Find frequent pairs where both tools are in population
        frequent_pairs = self.find_frequent_pairs()
        for t1_name, t2_name, prob in frequent_pairs:
            if t1_name in tool_patterns and t2_name in tool_patterns:
                pat1 = tool_patterns[t1_name]
                pat2 = tool_patterns[t2_name]
                
                # Avoid creating duplicates of existing composites
                existing = self._find_existing_composite(population, [t1_name, t2_name])
                if existing:
                    continue
                
                # Create composite pattern
                composite = CompositeToolPattern([pat1, pat2])
                composite.weight = 0.01
                
                # Insight boost proportional to co-occurrence probability
                composite.insight_boost = prob * 0.5
                
                return composite
        
        return None
    
    def _find_existing_composite(self, population: PatternPopulation, tool_names: List[str]) -> Optional[CompositeToolPattern]:
        """Check if a composite with exact same tool sequence already exists."""
        for p in population.patterns:
            if isinstance(p, CompositeToolPattern):
                seq = [pt.tool_name for pt in p.patterns]
                if seq == tool_names:
                    return p
        return None
