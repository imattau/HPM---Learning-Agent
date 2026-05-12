from hpm_ai_v6.hpm_model.storage.relation_pattern_emitter import RelationPatternEmitter
from hpm_ai_v6.hpm_model.core.cell import Cell
import numpy as np
import pytest


def make_cell(name, emb):
    return Cell(name=name, dim=0, embedding=emb)


def test_observe_returns_dim2_cell():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_cat", [1, 0, 0, 0])
    tgt = make_cell("word_sat", [0, 1, 0, 0])
    rel_cell = emitter.observe(src, "lexical_transition", tgt)
    assert rel_cell.dim == 2


def test_observe_name_is_rel_prefixed():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_cat", [1, 0, 0, 0])
    tgt = make_cell("word_sat", [0, 1, 0, 0])
    rel_cell = emitter.observe(src, "lexical_transition", tgt)
    assert rel_cell.name == "rel_lexical_transition"


def test_observe_embedding_converges_to_mean_offset():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_cat", [1, 0, 0, 0])
    tgt = make_cell("word_sat", [0, 1, 0, 0])
    for _ in range(50):
        emitter.observe(src, "lexical_transition", tgt)
    rel_cell = emitter.observe(src, "lexical_transition", tgt)
    expected = np.array([0, 1, 0, 0]) - np.array([1, 0, 0, 0])  # [-1,1,0,0]
    np.testing.assert_allclose(rel_cell.as_numpy(), expected, atol=0.1)


def test_same_relation_returns_same_cell_object():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1, 0, 0, 0])
    tgt = make_cell("word_b", [0, 1, 0, 0])
    cell1 = emitter.observe(src, "lexical_transition", tgt)
    cell2 = emitter.observe(src, "lexical_transition", tgt)
    assert cell1.name == cell2.name


def test_different_relations_produce_different_cells():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1, 0, 0, 0])
    tgt = make_cell("word_b", [0, 1, 0, 0])
    cell1 = emitter.observe(src, "lexical_transition", tgt)
    cell2 = emitter.observe(src, "causal_relation", tgt)
    assert cell1.name != cell2.name


def test_get_relation_cells_returns_all_observed():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1, 0, 0, 0])
    tgt = make_cell("word_b", [0, 1, 0, 0])
    emitter.observe(src, "lexical_transition", tgt)
    emitter.observe(src, "causal_relation", tgt)
    cells = emitter.get_relation_cells()
    names = [c.name for c, _ in cells]
    assert "rel_lexical_transition" in names
    assert "rel_causal_relation" in names


def test_get_relation_cells_weight_increases_with_observations():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1, 0, 0, 0])
    tgt = make_cell("word_b", [0, 1, 0, 0])
    for _ in range(5):
        emitter.observe(src, "lexical_transition", tgt)
    cells = emitter.get_relation_cells()
    weight = next(w for c, w in cells if c.name == "rel_lexical_transition")
    assert weight == pytest.approx(5.0)
