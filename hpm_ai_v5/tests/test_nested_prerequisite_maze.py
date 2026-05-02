from hpm_ai_v5.planning import NestedPrerequisiteMazePlanner


MAZE_A = [
    "S . . K1 # C",
    "# # . # # .",
    ". . . D1 . .",
    ". # # # . K2",
    ". T . . . D2",
    ". . . # # G",
]


MAZE_B = [
    "S . K1 . # .",
    ". # . . # C",
    ". . D1 . . .",
    "# . . # K2 .",
    ". T . . D2 .",
    ". . . # # G",
]


def test_nested_prerequisite_maze_prefers_full_chain_and_reuses_strategy() -> None:
    planner = NestedPrerequisiteMazePlanner()

    first = planner.solve(MAZE_A)
    second = planner.solve(MAZE_B)

    assert first.result == "success"
    assert first.selected_strategy == ["collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"]
    assert first.subgoals == ["collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"]
    assert first.rejected["goal_direct"] == "blocked by D2 without K2"
    assert first.rejected["coin_route"] == "short-term reward but increases distance and misses K2"
    assert first.rejected["trap_route"] == "terminal failure"
    assert first.rejected["avoid_trap_ignore_k2"] == "missing prerequisite chain"
    assert first.score_trace["goal_completion"] == 1.0
    assert first.score_trace["prerequisite_satisfaction"] == 1.0
    assert first.score_trace["safety"] == 1.0
    assert first.score_trace["subgoal_progress"] == 1.0
    assert first.score_trace["reusable_sequence_match"] > 0.0
    assert first.confidence > 0.0
    assert first.reasoning_trace["selected_strategy"] == ["collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"]
    assert first.reasoning_trace["subgoals"] == ["collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"]
    assert first.reasoning_trace["generation_trace"]["generated_strategy"] == ["collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"]
    assert "D1 blocks access to K2" in first.reasoning_trace["generation_trace"]["generation_reason"]
    assert first.reasoning_trace["generation_trace"]["affordances"]["ordered_milestones"] == ["K1", "D1", "K2", "D2", "G"]
    assert first.reasoning_trace["generation_trace"]["affordances"]["dependencies"] == {
        "K1": [],
        "D1": ["K1"],
        "K2": ["D1"],
        "D2": ["K2"],
        "G": ["D2"],
    }

    assert second.result == "success"
    assert second.selected_strategy == ["collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"]
    assert second.score_trace["strategy_reuse"] > 0.0
    assert second.reasoning_trace["selected_strategy"] == ["collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"]
    assert len(second.selected_plan) >= 5
