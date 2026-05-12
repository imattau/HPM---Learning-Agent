from types import SimpleNamespace

from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
from hpm_ai_v6.hpm_model.core.cell import Cell


class StubLearningAgent:
    def __init__(self, patterns, weights):
        self.patterns = list(patterns)
        self._weights = list(weights)

    def get_weights(self):
        return list(self._weights)

    def get_best_pattern(self):
        return self.patterns[0] if self.patterns else None


def test_maintenance_cycle_reports_loaded_and_improved_agents():
    reader = MultiAgentReader.__new__(MultiAgentReader)
    word_a = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    word_b = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    word_agent = StubLearningAgent(patterns=[Cell(name="w_alpha->beta", dim=1, embedding=[0.0, 1.0, 0.0], source=word_a, target=word_b)], weights=[0.8])
    ctx_agent = StubLearningAgent(patterns=[], weights=[])
    reader.agents = {
        "char": None,
        "word": word_agent,
        "contextual": ctx_agent,
        "phrase": None,
        "semantic": None,
        "causal": None,
    }

    def warm_start_from_cache(limit=None):
        return 2

    def train_sequence(sentences, enable_causal=False):
        new_pattern = Cell(name=f"w_{len(word_agent.patterns)}", dim=1, embedding=[0.0, 0.0, 1.0], source=word_a, target=word_b)
        word_agent.patterns.append(new_pattern)
        word_agent._weights.append(0.9)
        ctx_agent.patterns.append(Cell(name=f"ctx_{len(ctx_agent.patterns)}", dim=1, embedding=[0.0, 0.0, 1.0], source=word_a, target=word_b))
        ctx_agent._weights.append(0.7)

    reader.warm_start_from_cache = warm_start_from_cache
    reader.train_sequence = train_sequence

    report = MultiAgentReader.maintenance_cycle(
        reader,
        sentences=["Alice meets the rabbit."],
        hydrate_limit=2,
        retrain_epochs=1,
        enable_causal=False,
    )

    assert report["_summary"]["loaded"] == 2
    assert "word" in report["_summary"]["improved_agents"]
    assert "contextual" in report["_summary"]["improved_agents"]
    assert report["word"]["pattern_delta"] == 1
    assert report["contextual"]["pattern_delta"] == 1
