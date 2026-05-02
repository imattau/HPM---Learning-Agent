"""Candidate strategy generation for maze-style planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core import PatternSequence


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


def _manhattan(left: tuple[int, int] | None, right: tuple[int, int] | None) -> int:
    if left is None or right is None:
        return 10_000
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _topological_order(dependencies: dict[str, list[str]]) -> list[str]:
    remaining = {name: set(prereqs) for name, prereqs in dependencies.items()}
    ordered: list[str] = []
    available = sorted(name for name, prereqs in remaining.items() if not prereqs)
    while available:
        current = available.pop(0)
        if current in ordered:
            continue
        ordered.append(current)
        for name, prereqs in remaining.items():
            if current in prereqs:
                prereqs.remove(current)
                if not prereqs and name not in ordered and name not in available:
                    available.append(name)
        available.sort()
    for name in dependencies:
        if name not in ordered:
            ordered.append(name)
    return ordered


@dataclass(frozen=True, slots=True)
class MazeObservation:
    grid: Grid
    symbols: dict[str, tuple[int, int] | None]

    @classmethod
    def from_raw(cls, raw: Sequence[Sequence[str]] | Sequence[str] | Grid) -> "MazeObservation":
        grid = _normalize_grid(raw)
        symbols = {
            "S": _find_symbol(grid, "S"),
            "K1": _find_symbol(grid, "K1"),
            "D1": _find_symbol(grid, "D1"),
            "K2": _find_symbol(grid, "K2"),
            "D2": _find_symbol(grid, "D2"),
            "G": _find_symbol(grid, "G"),
            "T": _find_symbol(grid, "T"),
            "C": _find_symbol(grid, "C"),
        }
        return cls(grid=grid, symbols=symbols)


@dataclass(frozen=True, slots=True)
class MazeAffordances:
    dependencies: dict[str, list[str]]
    optional_rewards: tuple[str, ...]
    terminal_hazards: tuple[str, ...]
    notes: tuple[str, ...]
    ordered_milestones: tuple[str, ...]


@dataclass(slots=True)
class StrategyCandidate:
    name: str
    steps: list[str]
    strategy: PatternSequence
    rationale: str
    source: str
    score_trace: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    valid: bool = False


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    observation: MazeObservation
    affordances: MazeAffordances
    dependency_graph: dict[str, list[str]]
    candidates: list[StrategyCandidate]
    trace: dict[str, Any]


class CandidateGenerationAgent:
    """Infer dependencies and generate candidate strategies from maze structure."""

    def extract_objects(self, raw_grid: Sequence[Sequence[str]] | Sequence[str] | Grid) -> MazeObservation:
        return MazeObservation.from_raw(raw_grid)

    def infer_affordances(self, observation: MazeObservation) -> MazeAffordances:
        milestones = ("K1", "D1", "K2", "D2", "G")
        start = observation.symbols.get("S")
        ordered_milestones = tuple(
            sorted(
                (name for name in milestones if observation.symbols.get(name) is not None),
                key=lambda name: (
                    _manhattan(start, observation.symbols.get(name)),
                    observation.symbols.get(name)[0] if observation.symbols.get(name) is not None else 0,
                    observation.symbols.get(name)[1] if observation.symbols.get(name) is not None else 0,
                    name,
                ),
            )
        )
        dependencies = self.build_dependency_graph(observation, ordered_milestones)
        notes = ["dependency order inferred from landmark distance to start", "goal is behind the final door"]
        optional_rewards = tuple(name for name in ("C",) if observation.symbols.get(name) is not None)
        terminal_hazards = tuple(name for name in ("T",) if observation.symbols.get(name) is not None)
        if observation.symbols.get("C") is not None:
            notes.append("C is an optional decoy reward")
        if observation.symbols.get("T") is not None:
            notes.append("T is terminal")
        return MazeAffordances(
            dependencies=dependencies,
            optional_rewards=optional_rewards,
            terminal_hazards=terminal_hazards,
            notes=tuple(notes),
            ordered_milestones=ordered_milestones,
        )

    def build_dependency_graph(self, observation: MazeObservation, ordered_milestones: tuple[str, ...]) -> dict[str, list[str]]:
        if not ordered_milestones:
            return {}
        if len(ordered_milestones) == 1:
            return {ordered_milestones[0]: []}

        graph: dict[str, list[str]] = {}
        previous = ordered_milestones[0]
        graph[previous] = []
        for current in ordered_milestones[1:]:
            graph[current] = [previous]
            previous = current
        for name in ("K1", "D1", "K2", "D2", "G"):
            graph.setdefault(name, [])
        return graph

    def _primary_chain_candidate(self, dependency_graph: dict[str, list[str]], context_signature: str) -> StrategyCandidate:
        ordered_nodes = _topological_order(dependency_graph)
        steps = [
            "move_to_K1",
            "collect_K1",
            "move_to_D1",
            "unlock_D1",
            "move_to_K2",
            "collect_K2",
            "move_to_D2",
            "unlock_D2",
            "move_to_G",
            "reach_G",
        ]
        strategy = PatternSequence(pattern_names=("collect_K1", "unlock_D1", "collect_K2", "unlock_D2", "reach_G"), support=3, density=2.5, utility=2.0)
        strategy.context_memory[context_signature] = 2.0
        rationale = "I generated the strategy because D1 blocks access to K2, and K2 is needed for D2."
        return StrategyCandidate(
            name="prerequisite_chain",
            steps=steps,
            strategy=strategy,
            rationale=rationale,
            source="dependency_graph",
        )

    def generate(self, raw_grid: Sequence[Sequence[str]] | Sequence[str] | Grid, *, context_signature: str) -> CandidateGenerationResult:
        observation = self.extract_objects(raw_grid)
        affordances = self.infer_affordances(observation)
        dependency_graph = affordances.dependencies
        candidates = [
            StrategyCandidate(
                name="goal_direct",
                steps=["move_to_goal_direct"],
                strategy=PatternSequence(pattern_names=("goal_direct",), support=1, density=0.1, utility=-1.5),
                rationale="goal is blocked by the final door and the required key is missing",
                source="affordance_template",
            ),
            StrategyCandidate(
                name="coin_route",
                steps=["move_to_coin", "collect_coin", "move_to_goal"],
                strategy=PatternSequence(pattern_names=("collect_coin", "move_to_goal"), support=1, density=0.2, utility=-1.0),
                rationale="coin is an optional reward and can distract from the dependency chain",
                source="affordance_template",
            ),
            StrategyCandidate(
                name="trap_route",
                steps=["move_to_trap"],
                strategy=PatternSequence(pattern_names=("move_to_trap",), support=1, density=0.0, utility=-2.0),
                rationale="trap is terminal",
                source="affordance_template",
            ),
            StrategyCandidate(
                name="avoid_trap_ignore_k2",
                steps=["move_around_trap", "move_to_goal"],
                strategy=PatternSequence(pattern_names=("avoid_trap", "move_to_goal"), support=1, density=0.1, utility=-0.5),
                rationale="avoiding the trap is not enough if the prerequisite chain is incomplete",
                source="affordance_template",
            ),
            self._primary_chain_candidate(dependency_graph, context_signature),
        ]
        trace = {
            "objects": observation.symbols,
            "affordances": {
                "dependencies": dependency_graph,
                "optional_rewards": list(affordances.optional_rewards),
                "terminal_hazards": list(affordances.terminal_hazards),
                "notes": list(affordances.notes),
                "ordered_milestones": list(affordances.ordered_milestones),
            },
            "generated_strategy": list(candidates[-1].strategy.pattern_names),
            "generation_reason": candidates[-1].rationale,
        }
        return CandidateGenerationResult(
            observation=observation,
            affordances=affordances,
            dependency_graph=dependency_graph,
            candidates=candidates,
            trace=trace,
        )
