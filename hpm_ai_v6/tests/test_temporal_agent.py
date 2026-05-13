from types import SimpleNamespace

from hpm_ai_v6.agents.temporal_agent import TemporalAgent
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.core.temporal_cell import TemporalCell
from hpm_ai_v6.agents.causal_agent import CausalRule


def make_cell(name: str) -> Cell:
    return Cell(name=name, dim=0, embedding=[0.0, 0.0, 0.0])


def make_causal_rule(source: Cell, target: Cell, weight: float = 0.7) -> CausalRule:
    return CausalRule(
        name=f"causal_{source.name}_to_{target.name}",
        intervention=f"change {source.name}",
        effect_magnitude=weight,
        agent_impacted="word",
        source=source,
        target=target,
        metadata={"original_word": source.name.removeprefix("word_")},
    )


def test_temporal_cell_fields():
    cause = make_cell("word_rain")
    effect = make_cell("word_flood")
    cell = TemporalCell(
        name="temporal:rain->flood",
        dim=3,
        weight=0.8,
        embedding=[0.0, 0.0, 0.0],
        cause=cause,
        effect=effect,
        onset_weight=0.8,
        duration_weight=1.0,
    )

    assert cell.cause is cause
    assert cell.effect is effect
    assert cell.onset_weight == 0.8
    assert cell.duration_weight == 1.0
    assert cell.concurrent == []
    assert cell.lapsed is False


def test_segment_cells_from_causal_edges():
    agent = TemporalAgent()
    rain = make_cell("word_rain")
    flood = make_cell("word_flood")

    cells = agent.build_temporal_cells([make_causal_rule(rain, flood, weight=0.9)])

    assert len(cells) == 1
    cell = cells[0]
    assert cell.cause is rain
    assert cell.effect is flood
    assert cell.duration_weight == 1.0
    assert cell.onset_weight == 0.9


def test_spanning_cells_from_chain():
    agent = TemporalAgent()
    rain = make_cell("word_rain")
    flood = make_cell("word_flood")
    damage = make_cell("word_damage")

    cells = agent.build_temporal_cells(
        [
            make_causal_rule(rain, flood, weight=0.9),
            make_causal_rule(flood, damage, weight=0.7),
        ]
    )

    names = {(cell.cause.name, cell.effect.name) for cell in cells}
    assert ("word_rain", "word_flood") in names
    assert ("word_flood", "word_damage") in names
    assert ("word_rain", "word_damage") in names

    spanning = next(cell for cell in cells if cell.cause.name == "word_rain" and cell.effect.name == "word_damage")
    assert spanning.duration_weight == 2.0
    assert spanning.onset_weight == 0.7


def test_concurrent_detection():
    agent = TemporalAgent()
    rain = make_cell("word_rain")
    flood = make_cell("word_flood")
    mud = make_cell("word_mud")

    cells = agent.build_temporal_cells(
        [
            make_causal_rule(rain, flood, weight=0.9),
            make_causal_rule(rain, mud, weight=0.7),
        ]
    )

    flood_cell = next(cell for cell in cells if cell.effect.name == "word_flood")
    mud_cell = next(cell for cell in cells if cell.effect.name == "word_mud")
    assert mud_cell in flood_cell.concurrent
    assert flood_cell in mud_cell.concurrent


def test_lapsed_on_weight_decay():
    agent = TemporalAgent(threshold=0.5)
    rain = make_cell("word_rain")
    flood = make_cell("word_flood")

    agent.build_temporal_cells([make_causal_rule(rain, flood, weight=0.9)])
    agent.update_temporal_cells([make_causal_rule(rain, flood, weight=0.0)])
    agent.update_temporal_cells([make_causal_rule(rain, flood, weight=0.0)])

    cell = agent.temporal_index["word_rain"][0]
    assert cell.lapsed is True
    assert cell.onset_weight < 0.5


def test_update_preserves_lapsed_history():
    agent = TemporalAgent(threshold=0.5)
    rain = make_cell("word_rain")
    flood = make_cell("word_flood")
    damage = make_cell("word_damage")

    agent.build_temporal_cells([make_causal_rule(rain, flood, weight=0.9)])
    agent.update_temporal_cells([make_causal_rule(rain, flood, weight=0.0)])
    agent.update_temporal_cells([make_causal_rule(rain, flood, weight=0.0)])
    agent.update_temporal_cells([make_causal_rule(flood, damage, weight=0.8)])

    entries = agent.temporal_index["word_rain"]
    assert any(cell.lapsed for cell in entries)
    assert any(cell.effect.name == "word_flood" for cell in entries)

