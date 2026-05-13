from types import SimpleNamespace

from hpm_ai_v6.agents.causal_agent import CausalRule
from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
from hpm_ai_v6.agents.temporal_agent import TemporalAgent
from hpm_ai_v6.hpm_model.core.cell import Cell


class StubAgent:
    def __init__(self, patterns=None, weights=None, lookup=None):
        self.patterns = patterns or []
        self._weights = weights or []
        self._lookup = lookup or {}

    def get_weights(self):
        return list(self._weights)

    def _paging_lookup(self):
        return dict(self._lookup)


def make_word(name: str, embedding):
    return Cell(name=f"word_{name}", dim=0, embedding=embedding)


def make_edge(source: Cell, target: Cell, name: str | None = None, weight: float = 0.7):
    return Cell(
        name=name or f"w_{source.name}->{target.name}",
        dim=1,
        embedding=target.as_numpy() - source.as_numpy(),
        source=source,
        target=target,
        weight=weight,
    )


def build_reader():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    damage = make_word("damage", [0.0, 0.0, 1.0])

    causal_rule_1 = CausalRule(
        name="causal_rain_in_word",
        intervention="replace rain",
        effect_magnitude=0.9,
        agent_impacted="word",
        source=rain,
        target=flood,
        metadata={"original_word": "rain", "agent_impacted": "word"},
    )
    causal_rule_2 = CausalRule(
        name="causal_flood_in_word",
        intervention="replace flood",
        effect_magnitude=0.8,
        agent_impacted="word",
        source=flood,
        target=damage,
        metadata={"original_word": "flood", "agent_impacted": "word"},
    )

    word_agent = StubAgent(
        patterns=[
            make_edge(rain, flood, weight=0.9),
            make_edge(flood, damage, weight=0.8),
        ],
        weights=[0.9, 0.8],
        lookup={cell.name: cell for cell in [rain, flood, damage]},
    )
    causal_agent = StubAgent(
        patterns=[causal_rule_1, causal_rule_2],
        weights=[0.9, 0.8],
        lookup={causal_rule_1.source.name: causal_rule_1.source, causal_rule_1.target.name: causal_rule_1.target},
    )
    temporal_agent = TemporalAgent()
    temporal_agent.update_temporal_cells(causal_agent.patterns)

    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": causal_agent,
            "temporal": temporal_agent,
        }
    )
    return ReasoningAgent(reader)


def test_temporal_sequence_returns_ordered_intervals():
    agent = build_reader()

    cells = agent.temporal_sequence("rain")

    assert cells
    assert cells[0].duration_weight <= cells[-1].duration_weight


def test_temporal_between_follows_causal_chain():
    agent = build_reader()

    cells = agent.temporal_between("rain", "damage")

    assert len(cells) >= 2
    assert cells[0].cause.name == "word_rain"


def test_temporal_between_fallback_to_beam_search():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    damage = make_word("damage", [0.0, 0.0, 1.0])
    word_agent = StubAgent(
        patterns=[
            make_edge(rain, flood, weight=0.9),
            make_edge(flood, damage, weight=0.8),
        ],
        weights=[0.9, 0.8],
        lookup={cell.name: cell for cell in [rain, flood, damage]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
            "temporal": TemporalAgent(),
        }
    )
    agent = ReasoningAgent(reader)

    cells = agent.temporal_between("rain", "damage")

    assert len(cells) == 1
    assert cells[0].cause.name == "word_rain"
    assert cells[0].effect.name == "word_damage"
    assert cells[0].duration_weight == 2.0


def test_temporal_overlap_returns_concurrent_intervals():
    agent = build_reader()

    cells = agent.temporal_overlap("rain")

    assert cells
    assert any(cell.effect.name == "word_damage" for cell in cells)


def test_reason_with_trace_temporal_intent_parsing():
    agent = build_reader()

    trace = agent.reason_with_trace("When did rain lead to flood?")

    assert trace["intent"] == "temporal_sequence"
    assert trace["temporal_cells"]
    assert "temporal sequence" in trace["answer"].lower()


def test_include_lapsed_flag_exposes_history():
    agent = build_reader()
    temporal_agent = agent._temporal_agent()
    assert temporal_agent is not None
    rain_cells = temporal_agent.temporal_index.get("word_rain", [])
    if rain_cells:
        rain_cells[0].lapsed = True

    hidden = agent.temporal_sequence("rain", include_lapsed=False)
    shown = agent.temporal_sequence("rain", include_lapsed=True)

    assert len(shown) >= len(hidden)
