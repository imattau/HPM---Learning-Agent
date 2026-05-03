from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpm_ai_v5.arc import ArcSolver


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ARC-AGI-2" / "data" / "training"


@pytest.mark.parametrize(
    ("task_name", "expected_route"),
    [
        ("a79310a0.json", "translate_recolour"),
        ("b1948b0a.json", "identity_recolour"),
    ],
)
def test_arc_solver_on_real_arc_examples(task_name: str, expected_route: str) -> None:
    raw_task = json.loads((DATA_DIR / task_name).read_text())

    packet = ArcSolver().solve(raw_task)

    assert packet.context["arc"]["route"] == expected_route
    assert packet.context["arc"]["accepted"] is True
    assert packet.final_output == raw_task["test"][0]["output"]
    assert any(view["name"] == "image_polygraph" for view in packet.views)
    assert any(view["name"] == "arc_hypothesis" for view in packet.views)
    assert any(view["name"] == "arc_symmetry_completion" for view in packet.views)
    assert any(view["name"] == "arc_line_extension" for view in packet.views)
    assert any(view["name"] == "transformation_polygraph" for view in packet.views)
    assert any(view["name"] == "object_polygraph" and "relation_graph_4" in view for view in packet.views)
    assert packet.context["arc"]["edges"]
    assert packet.context["arc"]["hypotheses"]
    assert isinstance(packet.context["arc"]["symmetry_hypotheses"], list)
    assert isinstance(packet.context["arc"]["line_extension_hypotheses"], list)
    assert packet.context["arc"]["grid_deltas"]
    assert packet.context["arc"]["object_deltas"]
    assert packet.context["arc"]["colour_deltas"]
    assert packet.context["arc"]["structural_deltas"]
    assert any(entry["role"] == "adapter" for entry in packet.trace)
    assert any(entry["role"] == "agent" for entry in packet.trace)
