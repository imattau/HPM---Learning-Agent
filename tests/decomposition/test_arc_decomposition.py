
from __future__ import annotations

from hpm.decomposition import build_arc_decomposition_profile, extract_arc_decomposition, score_arc_decomposition


def test_arc_decomposition_extracts_explicit_parts():
    input_grid = [[1, 0], [0, 2]]
    output_grid = [[0, 1], [2, 0]]

    result = extract_arc_decomposition(input_grid, output_grid, task_id="task", pair_id="pair")

    assert result.coverage == 1.0
    assert result.part_count >= 2
    assert result.relation_count >= result.part_count
    assert result.composite_count >= 2
    assert result.trace.selected_pattern_ids
    assert result.unassigned_atom_ids == ()
    assert result.ambiguous_part_ids == ()

    profile = build_arc_decomposition_profile([result])
    score, breakdown = score_arc_decomposition(profile, result)
    assert score == 0.0
    assert breakdown["coverage_gap"] == 0.0
    assert breakdown["ambiguity_gap"] == 0.0


def test_arc_decomposition_marks_ambiguous_parts():
    input_grid = [[1, 0], [0, 1]]
    output_grid = [[0, 1], [1, 0]]

    result = extract_arc_decomposition(input_grid, output_grid, task_id="task", pair_id="ambiguous")

    assert result.coverage == 1.0
    assert result.ambiguous_part_ids
    assert result.trace.rejection_reason in {"low_support", "mixed_support"}
