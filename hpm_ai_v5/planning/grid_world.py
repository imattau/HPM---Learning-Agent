"""Nested key-door grid planning harness for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core import PatternEngine, PatternSequence, State


Grid = tuple[tuple[str, ...], ...]


def _normalize_grid(grid: Sequence[Sequence[str]] | Grid) -> Grid:
    return tuple(tuple(str(cell) for cell in row) for row in grid)


def _find_symbol(grid: Grid, symbol: str) -> tuple[int, int] | None:
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == symbol:
                return row_index, col_index
    return None


@dataclass(frozen=True, slots=True)
class GridWorldProblem:
    grid: Grid
    start: tuple[int, int]
    key: tuple[int, int]
    door: tuple[int, int]
    goal: tuple[int, int]
    trap: tuple[int, int] | None = None

    @classmethod
    def from_raw(cls, raw: Sequence[Sequence[str]] | Grid) -> "GridWorldProblem":
        grid = _normalize_grid(raw)
        start = _find_symbol(grid, "S")
        key = _find_symbol(grid, "K")
        door = _find_symbol(grid, "D")
        goal = _find_symbol(grid, "G")
        trap = _find_symbol(grid, "T")
        if start is None or key is None or door is None or goal is None:
            raise ValueError("Grid must contain S, K, D, and G")
        return cls(grid=grid, start=start, key=key, door=door, goal=goal, trap=trap)

    def context(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "key": self.key,
            "door": self.door,
            "goal": self.goal,
            "trap": self.trap,
            "door_status": "locked",
            "inventory": "empty",
            "mode": "nested_key_door",
        }

    def signature(self) -> str:
        trap = None if self.trap is None else f"{self.trap[0]}:{self.trap[1]}"
        return f"S={self.start}|K={self.key}|D={self.door}|G={self.goal}|T={trap}"


@dataclass(frozen=True, slots=True)
class PlanningResult:
    selected_plan: list[str]
    rejected_plans: dict[str, str]
    score_trace: dict[str, float]
    result: str
    reason: str
    selected_sequence: list[str] = field(default_factory=list)


@dataclass
class _CandidatePlan:
    name: str
    steps: list[str]
    sequence: PatternSequence
    score_trace: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    valid: bool = False


class GridWorldPlanner:
    """A small reasoning harness for deep planning."""

    def __init__(self, engine: PatternEngine | None = None) -> None:
        self.engine = engine or PatternEngine()

    def _candidates(self, problem: GridWorldProblem) -> list[_CandidatePlan]:
        context_signature = self.engine.store.context_signature(problem.context())
        candidates = [
            _CandidatePlan(
                name="greedy_shortest",
                steps=["move_to_goal_direct"],
                sequence=PatternSequence(pattern_names=("move_to_goal_direct",), support=1, density=0.1, utility=-0.5),
            ),
            _CandidatePlan(
                name="trap_route",
                steps=["move_to_trap", "move_to_goal"],
                sequence=PatternSequence(pattern_names=("move_to_trap", "move_to_goal"), support=1, density=0.1, utility=-1.0),
            ),
            _CandidatePlan(
                name="wander",
                steps=["move_random", "move_random", "move_random"],
                sequence=PatternSequence(pattern_names=("move_random", "move_random"), support=0, density=0.0, utility=-2.0),
            ),
            _CandidatePlan(
                name="key_first",
                steps=["move_to_key", "collect_key", "move_to_door", "unlock_door", "move_to_goal"],
                sequence=PatternSequence(
                    pattern_names=("collect_key", "unlock_door", "reach_goal"),
                    support=3,
                    density=2.0,
                    utility=1.0,
                ),
            ),
        ]
        for candidate in candidates:
            if candidate.name == "key_first":
                candidate.sequence.context_memory[context_signature] = 2.0
            elif candidate.name == "greedy_shortest":
                candidate.sequence.context_memory[context_signature] = 0.1
            elif candidate.name == "trap_route":
                candidate.sequence.context_memory[context_signature] = 0.0
            else:
                candidate.sequence.context_memory[context_signature] = 0.0
        return candidates

    def _simulate(self, candidate: _CandidatePlan, problem: GridWorldProblem) -> dict[str, float | str]:
        has_key = False
        door_open = False
        trapped = False
        reached_goal = False
        invalid = False
        reasons: list[str] = []

        for step in candidate.steps:
            if step == "move_to_trap":
                trapped = True
                reasons.append("trap risk")
                invalid = True
                break
            if step == "collect_key":
                has_key = True
            elif step == "unlock_door":
                if not has_key:
                    invalid = True
                    reasons.append("door requires key")
                else:
                    door_open = True
            elif step == "move_to_goal_direct" or step == "move_to_goal":
                if not door_open and problem.door is not None:
                    invalid = True
                    reasons.append("locked door without key")
                else:
                    reached_goal = True
            elif step == "move_random":
                pass

        goal_progress = 1.0 if reached_goal else 0.4 if door_open or has_key else 0.1
        prerequisite_satisfaction = 1.0 if has_key and door_open and reached_goal else 0.0
        safety = 0.0 if trapped else 1.0
        path_cost = len(candidate.steps) * 0.1
        trap_risk = 1.0 if trapped else 0.0
        invalid_action_penalty = 1.0 if invalid else 0.0
        sequence_density = candidate.sequence.score(context_signature=self.engine.store.context_signature(problem.context()), goal={"utility": 0.0, "beta": 1.0, "gamma": 0.5, "delta": 0.25})
        score = (
            goal_progress
            + prerequisite_satisfaction
            + safety
            + sequence_density
            - path_cost
            - trap_risk
            - invalid_action_penalty
        )
        reason = "success" if not invalid and reached_goal and has_key and door_open else (reasons[0] if reasons else "low goal efficiency")
        return {
            "goal_progress": goal_progress,
            "prerequisite_satisfaction": prerequisite_satisfaction,
            "safety": safety,
            "sequence_density": sequence_density,
            "path_cost": -path_cost,
            "trap_risk": -trap_risk,
            "invalid_action_penalty": -invalid_action_penalty,
            "score": score,
            "reason": reason,
        }

    def solve(self, raw_grid: Sequence[Sequence[str]] | Grid) -> PlanningResult:
        problem = GridWorldProblem.from_raw(raw_grid)
        context = problem.context()
        context_signature = self.engine.store.context_signature(context)
        self.engine.current_state = State(value=problem.signature(), context=context)
        self.engine.history = [self.engine.current_state]

        candidates = self._candidates(problem)
        self.engine.sequences = [candidate.sequence for candidate in candidates]
        selected_sequence = self.engine.select_sequence(goal={"beta": 1.0, "gamma": 3.0, "delta": 0.75, "utility": 1.0})

        best: _CandidatePlan | None = None
        rejected: dict[str, str] = {}
        for candidate in candidates:
            metrics = self._simulate(candidate, problem)
            candidate.score_trace = {key: float(value) for key, value in metrics.items() if key != "reason"}
            candidate.reason = str(metrics["reason"])
            candidate.valid = candidate.reason == "success"
            if selected_sequence is not None and tuple(selected_sequence.pattern_names) == tuple(candidate.sequence.pattern_names):
                candidate.score_trace["core_alignment"] = 1.0
                candidate.score_trace["score"] += 0.5
            else:
                candidate.score_trace["core_alignment"] = 0.0
            if not candidate.valid:
                rejected[candidate.name] = candidate.reason
            if best is None or candidate.score_trace["score"] > best.score_trace.get("score", float("-inf")):
                best = candidate

        if best is None:
            return PlanningResult(selected_plan=[], rejected_plans=rejected, score_trace={}, result="failure", reason="no candidates")

        selected_sequence_names = list(selected_sequence.pattern_names) if selected_sequence is not None else list(best.sequence.pattern_names)
        if selected_sequence is not None:
            best.sequence.context_memory.setdefault(context_signature, 0.0)

        return PlanningResult(
            selected_plan=best.steps,
            rejected_plans=rejected,
            score_trace=best.score_trace,
            result="success" if best.valid else "failure",
            reason=best.reason,
            selected_sequence=selected_sequence_names,
        )
