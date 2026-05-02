from hpm_ai_v5.planning import GridWorldPlanner


def test_nested_key_door_grid_prefers_prerequisite_respecting_sequence() -> None:
    grid = [
        "S..K",
        "##.#",
        "T..D",
        "...G",
    ]

    planner = GridWorldPlanner()
    result = planner.solve(grid)

    assert result.result == "success"
    assert result.selected_plan == ["move_to_key", "collect_key", "move_to_door", "unlock_door", "move_to_goal"]
    assert result.selected_sequence == ["collect_key", "unlock_door", "reach_goal"]
    assert result.rejected_plans["greedy_shortest"] == "locked door without key"
    assert result.rejected_plans["trap_route"] == "trap risk"
    assert result.rejected_plans["wander"] == "low goal efficiency"
    assert result.score_trace["goal_progress"] == 1.0
    assert result.score_trace["prerequisite_satisfaction"] == 1.0
    assert result.score_trace["safety"] == 1.0
    assert result.score_trace["path_cost"] < 0.0
    assert result.score_trace["core_alignment"] == 1.0
    assert len(result.selected_plan) >= 5
