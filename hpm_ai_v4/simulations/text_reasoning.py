# hpm_ai_v4/simulations/text_reasoning.py
import numpy as np
from typing import List, Tuple, Optional
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.agents.reasoning import Reasoner
from hpm_ai_v4.pattern import HierarchicalPattern

CLASS_NAMES = ['letter', 'digit', 'space', 'punctuation', 'newline']
SPACE_CLASS_ID = 2


class TextReasoningInterface:
    def __init__(self,
                 L1_patterns: List[HierarchicalPattern], L1_reasoner: Reasoner,
                 L2_patterns: List[HierarchicalPattern], L2_reasoner: Optional[Reasoner],
                 L3_patterns: List[HierarchicalPattern], L3_reasoner: Optional[Reasoner]):
        self.L1_patterns = L1_patterns
        self.L2_patterns = L2_patterns
        self.L3_patterns = L3_patterns
        self.L1_reasoner = L1_reasoner
        self.L2_reasoner = L2_reasoner
        self.L3_reasoner = L3_reasoner
        self._adapter = CharClassAdapter()

    def _encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            if ch == '\n':
                ids.append(self._adapter.encode(-22))
            elif 32 <= ord(ch) <= 126:
                ids.append(self._adapter.encode(ord(ch) - 32))
        return ids

    def next_char_predict(self, prefix_str: str) -> List[Tuple[str, float]]:
        obs_seq = self._encode(prefix_str)
        relevant = self.L1_reasoner.get_relevant_patterns(obs_seq, top_k=5)
        dist = self.L1_reasoner.compose_predictions(relevant, obs_seq)  # shape (5,)
        dist = dist / (dist.sum() + 1e-12)
        top5 = np.argsort(dist)[::-1][:5]
        return [(CLASS_NAMES[i], float(dist[i])) for i in top5]

    def word_boundary_predict(self, prefix_str: str) -> float:
        if self.L2_reasoner is None or not self.L2_patterns:
            return 0.0
        obs_seq = self._encode(prefix_str)
        # Get L1 latent sequence for L2 input
        if not self.L1_patterns:
            return 0.0
        best_L1 = max(self.L1_patterns, key=lambda p: p.weight)
        l1_states = [best_L1.get_top_state(obs_seq[max(0, i-20):i+1])
                     for i in range(len(obs_seq))]
        relevant = self.L2_reasoner.get_relevant_patterns(l1_states, top_k=3)
        dist = self.L2_reasoner.compose_predictions(relevant, l1_states)
        dist = dist / (dist.sum() + 1e-12)
        # State 0 is taken as word-boundary by convention (highest weight pattern)
        return float(dist[0])

    def plan_to_space(self, horizon: int, num_rollouts: int) -> List[str]:
        if self.L3_reasoner is None:
            return []
        seq = self.L3_reasoner.plan(goal_state=SPACE_CLASS_ID,
                                    horizon=horizon, num_rollouts=num_rollouts)
        return [CLASS_NAMES[min(s, 4)] for s in seq]

    def counterfactual_shift(self, context: str, forced_class: int) -> List[Tuple[str, float]]:
        obs_seq = self._encode(context)
        relevant = self.L1_reasoner.get_relevant_patterns(obs_seq, top_k=5)
        blended = np.zeros(5)
        total_w = sum(p.weight for p in relevant) + 1e-12
        for p in relevant:
            _, intervened = self.L1_reasoner.counterfactual(p, obs_seq, forced_class)
            blended += (p.weight / total_w) * intervened[:5]
        blended = blended / (blended.sum() + 1e-12)
        top5 = np.argsort(blended)[::-1][:5]
        return [(CLASS_NAMES[i], float(blended[i])) for i in top5]
