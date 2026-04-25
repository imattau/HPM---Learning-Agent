"""LayeredAgent: L1 (char classes, obs_dim=5) + L2 (raw chars, obs_dim=95)."""
from typing import List, Tuple, Dict, Any
import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern


def _init_equal_weights(agent: HPMAgent, hier_k: int, obs_dim: int) -> None:
    """Replace agent.patterns with equal-weight hier+flat population."""
    agent.patterns = []
    # 4 Hierarchical patterns
    for i in range(4):
        p = HierarchicalPattern(i, latent_dim=hier_k, obs_dim=obs_dim)
        p.weight = 0.15
        agent.patterns.append(p)
    # 2 Flat patterns
    for i in range(4, 6):
        p = FlatPattern.flat(i, obs_dim=obs_dim)
        p.weight = 0.10
        agent.patterns.append(p)


class LayeredAgent:
    """Two-level HPM agent: L1 learns char-class structure, L2 learns actual chars."""

    def __init__(self, num_workers: int = 1):
        self._adapter = CharClassAdapter()
        self.l1 = HPMAgent(obs_dim=5, num_initial_patterns=4, num_workers=num_workers)
        self.l2 = HPMAgent(obs_dim=95, num_initial_patterns=4, num_workers=num_workers)
        _init_equal_weights(self.l1, hier_k=2, obs_dim=5)
        _init_equal_weights(self.l2, hier_k=4, obs_dim=95)

    def perceive(self, raw_char_id: int) -> None:
        """Feed one character to both levels."""
        class_id = self._adapter.encode(raw_char_id)
        self.l1.perceive_and_learn(class_id)
        self.l2.perceive_and_learn(raw_char_id)

    def generate(self, steps: int = 80) -> str:
        """Sample from L2 and decode to printable characters."""
        future = self.l2.reasoner.simulate_future(steps=steps, top_k=3)
        return "".join(chr(v + 32) for v in future if 0 <= v <= 94)

    def predict_next_chars(self, context_raw: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
        """Top-k next character predictions from L2."""
        relevant = self.l2.reasoner.get_relevant_patterns(context_raw, top_k=top_k)
        if not relevant:
            return []
        dist = self.l2.reasoner.compose_predictions(relevant, context_raw)
        top = np.argsort(dist)[::-1][:top_k]
        return [(chr(i + 32), float(dist[i])) for i in top]

    def l1_metrics(self) -> Dict[str, Any]:
        """Summary metrics for L1 population."""
        top3 = sorted(self.l1.patterns, key=lambda p: -p.weight)[:3]
        mi = float(np.mean([p.compression() for p in top3])) if top3 else 0.0
        return {
            'pop_size': len(self.l1.patterns),
            'mi': mi,
            'stage': self.l1.development.level,
            'best_weight': max(p.weight for p in self.l1.patterns) if self.l1.patterns else 0.0,
        }

    def l2_metrics(self, recent_raw: List[int]) -> Dict[str, Any]:
        """Prediction accuracy of L2 over recent char buffer."""
        correct = 0
        total = max(1, len(recent_raw) - 1)
        for i in range(len(recent_raw) - 1):
            ctx = recent_raw[max(0, i - 20):i]
            actual = recent_raw[i + 1]
            relevant = self.l2.reasoner.get_relevant_patterns(ctx, top_k=3)
            if relevant:
                dist = self.l2.reasoner.compose_predictions(relevant, ctx)
                if int(np.argmax(dist)) == actual:
                    correct += 1
        return {
            'accuracy': correct / total,
            'pop_size': len(self.l2.patterns),
        }
