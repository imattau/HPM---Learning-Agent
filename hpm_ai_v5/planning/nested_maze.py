"""Nested prerequisite maze planning harness for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core import PatternEngine, State
from .candidate_generation import CandidateGenerationAgent, StrategyCandidate


Grid = tuple[tuple[str, ...], ...]


def _normalize_row(row: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(row, str):
        stripped = row.strip()
        if " " in stripped:
            return tuple(token for token in stripped.split() if token)
        return tuple(stripped)
    return tuple(str(cell) for cell in row)


def _normalize_grid(grid: Sequence[Sequence[str]] | Sequence[str] | Grid) -> Grid:
    return tuple(_normalize_row(row) for row in grid)


def _find_symbol(grid: Grid, symbol: str) -> tuple[int, int] | None:
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == symbol:
                return row_index, col_index
    return None


@dataclass(frozen=True, slots=True)
class NestedMazeProblem:
    grid: Grid
    start: tuple[int, int]
    key1: tuple[int, int]
    door1: tuple[int, int]
    key2: tuple[int, int]
    door2: tuple[int, int]
    goal: tuple[int, int]
    trap: tuple[int, int] | None = None
    coin: tuple[int, int] | None = None

    @classmethod
    def from_raw(cls, raw: Sequence[Sequence[str]] | Sequence[str] | Grid) -> "NestedMazeProblem":
        grid = _normalize_grid(raw)
        start = _find_symbol(grid, "S")
        key1 = _find_symbol(grid, "K1")
        door1 = _find_symbol(grid, "D1")
        key2 = _find_symbol(grid, "K2")
        door2 = _find_symbol(grid, "D2")
        goal = _find_symbol(grid, "G")
        trap = _find_symbol(grid, "T")
        coin = _find_symbol(grid, "C")
        if start is None or key1 is None or door1 is None or key2 is None or door2 is None or goal is None:
            raise ValueError("Grid must contain S, K1, D1, K2, D2, and G")
        return cls(grid=grid, start=start, key1=key1, door1=door1, key2=key2, door2=door2, goal=goal, trap=trap, coin=coin)

    def dependency_chain(self) -> tuple[str, ...]:
        return ("collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G")

    def context(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "key1": self.key1,
            "door1": self.door1,
            "key2": self.key2,
            "door2": self.door2,
            "goal": self.goal,
            "trap": self.trap,
            "coin": self.coin,
            "mode": "nested_prerequisite_maze",
            "dependency_signature": "|".join(self.dependency_chain()),
        }

    def signature(self) -> str:
        trap = None if self.trap is None else f"{self.trap[0]}:{self.trap[1]}"
        coin = None if self.coin is None else f"{self.coin[0]}:{self.coin[1]}"
        return (
            f"S={self.start}|K1={self.key1}|D1={self.door1}|"
            f"K2={self.key2}|D2={self.door2}|G={self.goal}|T={trap}|C={coin}"
        )


@dataclass(frozen=True, slots=True)
class MazePlanningResult:
    selected_strategy: list[str]
    selected_plan: list[str]
    rejected: dict[str, str]
    subgoals: list[str]
    score_trace: dict[str, float]
    result: str
    reason: str
    confidence: float
    reasoning_trace: dict[str, Any]


class NestedPrerequisiteMazePlanner:
    """A shallow but reusable strategy planner for nested prerequisite mazes."""

    def __init__(self, engine: PatternEngine | None = None) -> None:
        self.engine = engine or PatternEngine()
        self.strategy_memory: dict[str, float] = {}
        self.generator = CandidateGenerationAgent()

    def _strategy_signature(self, strategy) -> str:
        return "|".join(strategy.canonical_names())

    def _simulate(self, candidate: StrategyCandidate, problem: NestedMazeProblem) -> dict[str, float | str]:
        has_k1 = False
        d1_open = False
        has_k2 = False
        d2_open = False
        trapped = False
        reached_goal = False
        coin_taken = False
        invalid = False
        reasons: list[str] = []
        completed_subgoals = 0

        for step in candidate.steps:
            if step == "move_to_trap":
                trapped = True
                invalid = True
                reasons.append("terminal failure")
                break
            if step == "collect_K1":
                has_k1 = True
                completed_subgoals = max(completed_subgoals, 1)
            elif step == "unlock_D1":
                if not has_k1:
                    invalid = True
                    reasons.append("D1 requires K1")
                else:
                    d1_open = True
                    completed_subgoals = max(completed_subgoals, 2)
            elif step == "collect_K2":
                if not d1_open:
                    invalid = True
                    reasons.append("K2 unreachable before D1")
                else:
                    has_k2 = True
                    completed_subgoals = max(completed_subgoals, 3)
            elif step == "unlock_D2":
                if not has_k2:
                    invalid = True
                    reasons.append("D2 requires K2")
                else:
                    d2_open = True
                    completed_subgoals = max(completed_subgoals, 4)
            elif step == "collect_coin":
                coin_taken = True
            elif step == "reach_G" or step == "move_to_goal":
                if not d2_open and problem.door2 is not None:
                    invalid = True
                    reasons.append("goal blocked by D2 without K2")
                else:
                    reached_goal = True
                    completed_subgoals = 5
            elif step == "move_to_goal_direct":
                if not d2_open:
                    invalid = True
                    reasons.append("blocked by D2 without K2")
                else:
                    reached_goal = True
            elif step == "move_random" or step.startswith("move_"):
                pass

        goal_completion = 1.0 if reached_goal else 0.65 if d2_open else 0.35 if d1_open else 0.1
        prerequisite_satisfaction = 1.0 if reached_goal and has_k1 and d1_open and has_k2 and d2_open else completed_subgoals / 5.0
        safety = 0.0 if trapped else 1.0
        subgoal_progress = completed_subgoals / 5.0
        sequence_match = candidate.strategy.score(
            context_signature=self.engine.store.context_signature(problem.context()),
            goal={"alpha": 0.0, "beta": 0.8, "gamma": 0.7, "delta": 0.4, "utility": candidate.strategy.utility},
        )
        path_cost = len(candidate.steps) * 0.08
        trap_risk = 1.0 if trapped else 0.0
        dead_end_penalty = 1.0 if coin_taken and not reached_goal else 0.0
        decoy_reward_overfitting = 0.5 if coin_taken and not has_k2 else 0.0
        score = (
            goal_completion
            + prerequisite_satisfaction
            + safety
            + subgoal_progress
            + sequence_match
            - path_cost
            - trap_risk
            - dead_end_penalty
            - decoy_reward_overfitting
        )
        if not reasons:
            if coin_taken and not reached_goal:
                reasons.append("short-term reward but increases distance and misses K2")
            elif invalid and not reached_goal:
                reasons.append("missing prerequisite chain")
            elif not reached_goal:
                reasons.append("low goal efficiency")
        if not invalid and reached_goal and has_k1 and d1_open and has_k2 and d2_open:
            reason = "success"
        elif candidate.name == "trap_route":
            reason = "terminal failure"
        elif candidate.name == "coin_route":
            reason = "short-term reward but increases distance and misses K2"
        elif candidate.name == "goal_direct":
            reason = "blocked by D2 without K2"
        elif candidate.name == "avoid_trap_ignore_k2":
            reason = "missing prerequisite chain"
        else:
            reason = reasons[0]
        return {
            "goal_completion": goal_completion,
            "prerequisite_satisfaction": prerequisite_satisfaction,
            "safety": safety,
            "subgoal_progress": subgoal_progress,
            "reusable_sequence_match": sequence_match,
            "path_cost": -path_cost,
            "trap_risk": -trap_risk,
            "dead_end_penalty": -dead_end_penalty,
            "decoy_reward_overfitting": -decoy_reward_overfitting,
            "score": score,
            "reason": reason,
        }

    def solve(self, raw_grid: Sequence[Sequence[str]] | Sequence[str] | Grid) -> MazePlanningResult:
        problem = NestedMazeProblem.from_raw(raw_grid)
        context = problem.context()
        context_signature = self.engine.store.context_signature(context)
        self.engine.current_state = State(value=problem.signature(), context=context)
        self.engine.history = [self.engine.current_state]

        generation = self.generator.generate(raw_grid, context_signature=context_signature)
        candidates = list(generation.candidates)
        self.engine.sequences = [candidate.strategy for candidate in candidates]
        selected_sequence = self.engine.select_sequence(goal={"beta": 1.0, "gamma": 3.0, "delta": 0.75, "utility": 1.0})

        best: StrategyCandidate | None = None
        rejected: dict[str, str] = {}
        for candidate in candidates:
            metrics = self._simulate(candidate, problem)
            candidate.score_trace = {key: float(value) for key, value in metrics.items() if key != "reason"}
            candidate.reason = str(metrics["reason"])
            candidate.valid = candidate.reason == "success"
            sig = self._strategy_signature(candidate.strategy)
            if candidate.name == "prerequisite_chain" and sig in self.strategy_memory:
                candidate.score_trace["strategy_reuse"] = self.strategy_memory[sig]
                candidate.score_trace["score"] += self.strategy_memory[sig]
            else:
                candidate.score_trace["strategy_reuse"] = 0.0
            if selected_sequence is not None and tuple(selected_sequence.pattern_names) == tuple(candidate.strategy.pattern_names):
                candidate.score_trace["core_alignment"] = 1.0
                candidate.score_trace["score"] += 0.5
            else:
                candidate.score_trace["core_alignment"] = 0.0
            if not candidate.valid:
                rejected[candidate.name] = candidate.reason
            if best is None or candidate.score_trace["score"] > best.score_trace.get("score", float("-inf")):
                best = candidate

        if best is None:
            return MazePlanningResult(
                selected_strategy=[],
                selected_plan=[],
                rejected=rejected,
                subgoals=list(problem.dependency_chain()),
                score_trace={},
                result="failure",
                reason="no candidates",
                confidence=0.0,
                reasoning_trace={"generation_trace": generation.trace},
            )

        selected_strategy = list(best.strategy.pattern_names)
        selected_sequence_names = list(selected_sequence.pattern_names) if selected_sequence is not None else selected_strategy
        if best.valid:
            sig = self._strategy_signature(best.strategy)
            self.strategy_memory[sig] = self.strategy_memory.get(sig, 0.0) + 1.0

        confidence = max(0.0, min(1.0, best.score_trace["score"] / (abs(best.score_trace["score"]) + 1.0)))
        reasoning_trace = {
            "observations": [state.value for state in self.engine.history[-3:]],
            "candidate_sequences": [
                {
                    "name": candidate.name,
                    "pattern_names": list(candidate.strategy.pattern_names),
                    "score": candidate.score_trace.get("score", 0.0),
                    "reason": candidate.reason,
                }
                for candidate in candidates
            ],
            "rejected": rejected,
            "selected_strategy": selected_strategy,
            "subgoals": list(problem.dependency_chain()),
            "score_trace": best.score_trace,
            "confidence": confidence,
            "generation_trace": generation.trace,
        }

        return MazePlanningResult(
            selected_strategy=selected_strategy,
            selected_plan=best.steps,
            rejected=rejected,
            subgoals=list(problem.dependency_chain()),
            score_trace=best.score_trace,
            result="success" if best.valid else "failure",
            reason=best.reason,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
        )
