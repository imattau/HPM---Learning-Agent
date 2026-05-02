from hpm_ai_v5.planning import CompositionalTransformationWorldPlanner


CTW_A = [
    "S . X . D",
    "# # . # .",
    "A . T . B",
    ". . . . T",
    ". . . . G",
]


CTW_B = [
    "S . . X .",
    ". # . . D",
    "A . T . .",
    ". . B . T",
    ". . . . G",
]


def test_ctw_discovers_rules_and_reuses_them_in_a_new_layout() -> None:
    planner = CompositionalTransformationWorldPlanner()

    first = planner.solve(CTW_A)
    second = planner.solve(CTW_B)

    assert first["result"] == "success"
    assert first["selected_strategy"] == ["activate_X", "open_D", "activate_A", "use_T", "reach_G"]
    assert first["rejected"]["goal_direct"] == "goal blocked without discovered rules"
    assert first["rejected"]["ignore_toggle"] == "teleporter requires A"
    assert first["score_trace"]["rule_accuracy"] > 0.0
    assert first["score_trace"]["plan_success"] == 1.0
    assert first["reasoning_trace"]["selected_strategy"] == ["activate_X", "open_D", "activate_A", "use_T", "reach_G"]
    assert first["reasoning_trace"]["generation_trace"]["interactions"][0]["source"] == "X"
    assert first["reasoning_trace"]["generation_trace"]["interactions"][0]["target"] == "D"
    assert first["reasoning_trace"]["generation_trace"]["interactions"][0]["evidence"].startswith("X and D co-occur")
    assert first["reasoning_trace"]["generation_trace"]["rules"]
    assert first["reasoning_trace"]["generation_trace"]["rules"][0]["rationale"] == "X toggles D → therefore activate X before door"
    assert first["reasoning_trace"]["generation_trace"]["rules"][1]["rationale"] == "A enables T → therefore activate A before teleporting"

    assert second["result"] == "success"
    assert second["selected_strategy"] == ["activate_X", "open_D", "activate_A", "use_T", "reach_G"]
    assert second["score_trace"]["rule_reuse"] > 0.0
    assert second["reasoning_trace"]["selected_strategy"] == ["activate_X", "open_D", "activate_A", "use_T", "reach_G"]
    assert second["reasoning_trace"]["generation_trace"]["interactions"][1]["source"] == "A"
    assert second["reasoning_trace"]["generation_trace"]["interactions"][1]["target"] == "T"
    assert second["reasoning_trace"]["generation_trace"]["interactions"][1]["evidence"].startswith("A co-occurs")
    assert len(second["selected_plan"]) >= 5
