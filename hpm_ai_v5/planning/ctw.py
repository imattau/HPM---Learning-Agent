"""Compositional Transformation World planning harness for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core import PatternEngine, PatternSequence, State


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


def _find_first(grid: Grid, symbol: str) -> tuple[int, int] | None:
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == symbol:
                return row_index, col_index
    return None


def _find_all(grid: Grid, symbol: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell == symbol
    )


def _manhattan(left: tuple[int, int] | None, right: tuple[int, int] | None) -> int:
    if left is None or right is None:
        return 10_000
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


@dataclass(frozen=True, slots=True)
class CTWObservation:
    grid: Grid
    symbols: dict[str, tuple[int, int] | tuple[tuple[int, int], ...] | None]

    @classmethod
    def from_raw(cls, raw: Sequence[Sequence[str]] | Sequence[str] | Grid) -> "CTWObservation":
        grid = _normalize_grid(raw)
        symbols: dict[str, tuple[int, int] | tuple[tuple[int, int], ...] | None] = {
            "S": _find_first(grid, "S"),
            "G": _find_first(grid, "G"),
            "A": _find_first(grid, "A"),
            "B": _find_first(grid, "B"),
            "X": _find_first(grid, "X"),
            "D": _find_first(grid, "D"),
            "T": _find_all(grid, "T"),
        }
        return cls(grid=grid, symbols=symbols)


@dataclass(frozen=True, slots=True)
class CTWInteraction:
    source: str
    relation: str
    target: str
    evidence: str
    confidence: float


@dataclass(frozen=True, slots=True)
class CTWRule:
    name: str
    antecedent: str
    consequent: str
    rationale: str
    confidence: float


@dataclass(frozen=True, slots=True)
class CTWDiscoveryResult:
    observation: CTWObservation
    interactions: tuple[CTWInteraction, ...]
    rules: tuple[CTWRule, ...]
    dependency_graph: dict[str, list[str]]
    trace: dict[str, Any]


@dataclass(slots=True)
class CTWCandidate:
    name: str
    steps: list[str]
    strategy: PatternSequence
    rationale: str
    source: str
    score_trace: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    valid: bool = False


class CTWDiscoveryAgent:
    """Discover transformation rules from compositional grid structure."""

    def extract_objects(self, raw_grid: Sequence[Sequence[str]] | Sequence[str] | Grid) -> CTWObservation:
        return CTWObservation.from_raw(raw_grid)

    def discover_interactions(self, observation: CTWObservation) -> tuple[CTWInteraction, ...]:
        start = observation.symbols.get("S")
        x = observation.symbols.get("X")
        d = observation.symbols.get("D")
        a = observation.symbols.get("A")
        t = observation.symbols.get("T")
        b = observation.symbols.get("B")
        interactions: list[CTWInteraction] = []

        if x is not None and d is not None:
            confidence = 0.95 if _manhattan(start, x) <= _manhattan(start, d) else 0.8
            interactions.append(
                CTWInteraction(
                    source="X",
                    relation="influences",
                    target="D",
                    evidence="X and D co-occur with X closer to the start",
                    confidence=confidence,
                )
            )
        if a is not None and t is not None:
            confidence = 0.9 if len(t) >= 1 else 0.7
            interactions.append(
                CTWInteraction(
                    source="A",
                    relation="enables",
                    target="T",
                    evidence="A co-occurs with paired teleporters",
                    confidence=confidence,
                )
            )
        if t is not None and len(t) >= 2:
            interactions.append(
                CTWInteraction(
                    source="T",
                    relation="shortcuts_to",
                    target="G",
                    evidence="paired teleporters are present",
                    confidence=0.85,
                )
            )
        if b is not None:
            interactions.append(
                CTWInteraction(
                    source="B",
                    relation="blocks",
                    target="A",
                    evidence="B is present as a blocking feature",
                    confidence=0.75,
                )
            )
        return tuple(interactions)

    def discover_rules(self, interactions: Sequence[CTWInteraction]) -> tuple[CTWRule, ...]:
        rules: list[CTWRule] = []
        for interaction in interactions:
            if interaction.source == "X" and interaction.target == "D":
                rules.append(
                    CTWRule(
                        name="toggle_door",
                        antecedent="X",
                        consequent="D",
                        rationale="X toggles D → therefore activate X before door",
                        confidence=interaction.confidence,
                    )
                )
            elif interaction.source == "A" and interaction.target == "T":
                rules.append(
                    CTWRule(
                        name="enable_teleporter",
                        antecedent="A",
                        consequent="T",
                        rationale="A enables T → therefore activate A before teleporting",
                        confidence=interaction.confidence,
                    )
                )
            elif interaction.source == "T" and interaction.target == "G":
                rules.append(
                    CTWRule(
                        name="teleporter_shortcut",
                        antecedent="T",
                        consequent="G",
                        rationale="T moves the player to the paired T and shortcuts the route to G",
                        confidence=interaction.confidence,
                    )
                )
            elif interaction.source == "B" and interaction.target == "A":
                rules.append(
                    CTWRule(
                        name="blocker_control",
                        antecedent="B",
                        consequent="A",
                        rationale="B blocks movement unless A is activated",
                        confidence=interaction.confidence,
                    )
                )
        return tuple(rules)

    def build_dependency_graph(self, interactions: Sequence[CTWInteraction]) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        if any(interaction.source == "X" and interaction.target == "D" for interaction in interactions):
            graph["open_D"] = ["activate_X"]
        if any(interaction.source == "A" and interaction.target == "T" for interaction in interactions):
            graph["enable_T"] = ["activate_A"]
        if any(interaction.source == "T" and interaction.target == "G" for interaction in interactions):
            graph["reach_G"] = ["use_T"]
        if any(interaction.source == "B" and interaction.target == "A" for interaction in interactions):
            graph["avoid_B"] = ["activate_A"]
        graph.setdefault("activate_X", [])
        graph.setdefault("activate_A", [])
        graph.setdefault("use_T", graph.get("enable_T", ["activate_A"]))
        graph.setdefault("reach_G", graph.get("reach_G", []))
        return graph

    def generate(self, raw_grid: Sequence[Sequence[str]] | Sequence[str] | Grid, *, context_signature: str) -> CTWDiscoveryResult:
        observation = self.extract_objects(raw_grid)
        interactions = self.discover_interactions(observation)
        rules = self.discover_rules(interactions)
        dependency_graph = self.build_dependency_graph(interactions)
        trace = {
            "objects": observation.symbols,
            "interactions": [
                {
                    "source": interaction.source,
                    "relation": interaction.relation,
                    "target": interaction.target,
                    "evidence": interaction.evidence,
                    "confidence": interaction.confidence,
                }
                for interaction in interactions
            ],
            "rules": [
                {
                    "name": rule.name,
                    "antecedent": rule.antecedent,
                    "consequent": rule.consequent,
                    "rationale": rule.rationale,
                    "confidence": rule.confidence,
                }
                for rule in rules
            ],
            "dependency_graph": dependency_graph,
            "context_signature": context_signature,
        }
        return CTWDiscoveryResult(
            observation=observation,
            interactions=interactions,
            rules=rules,
            dependency_graph=dependency_graph,
            trace=trace,
        )


class CompositionalTransformationWorldPlanner:
    """Plan by discovering transformation rules and then scoring trajectories."""

    def __init__(self, engine: PatternEngine | None = None) -> None:
        self.engine = engine or PatternEngine()
        self.discovery = CTWDiscoveryAgent()
        self.rule_memory: dict[str, float] = {}

    def _rule_signature(self, rules: Sequence[CTWRule]) -> str:
        return "|".join(f"{rule.antecedent}->{rule.consequent}" for rule in rules)

    def _candidate_pool(self, discovery: CTWDiscoveryResult, context_signature: str) -> list[CTWCandidate]:
        sequence = PatternSequence(pattern_names=("activate_X", "open_D", "activate_A", "use_T", "reach_G"), support=2, density=2.0, utility=2.0)
        sig = self._rule_signature(discovery.rules)
        sequence.context_memory[context_signature] = 2.0 + self.rule_memory.get(sig, 0.0)
        generated = CTWCandidate(
            name="discovered_composition",
            steps=["activate_X", "open_D", "activate_A", "use_T", "reach_G"],
            strategy=sequence,
            rationale="; ".join(interaction.evidence for interaction in discovery.interactions) if discovery.interactions else "no interactions discovered",
            source="rule_discovery",
        )
        return [
            CTWCandidate(
                name="goal_direct",
                steps=["move_to_goal_direct"],
                strategy=PatternSequence(pattern_names=("goal_direct",), support=1, density=0.1, utility=-1.5),
                rationale="goal is blocked before the transformation rules are applied",
                source="baseline",
            ),
            CTWCandidate(
                name="ignore_toggle",
                steps=["use_T", "reach_G"],
                strategy=PatternSequence(pattern_names=("use_T", "reach_G"), support=1, density=0.1, utility=-0.5),
                rationale="teleporter use without activation misses the required transformation",
                source="baseline",
            ),
            CTWCandidate(
                name="ignore_teleporter",
                steps=["activate_X", "open_D", "reach_G"],
                strategy=PatternSequence(pattern_names=("activate_X", "open_D", "reach_G"), support=1, density=0.1, utility=-0.5),
                rationale="door toggle alone is insufficient if the teleporter rule is discovered",
                source="baseline",
            ),
            generated,
        ]

    def _simulate(self, candidate: CTWCandidate, discovery: CTWDiscoveryResult) -> dict[str, float | str]:
        door_open = False
        teleporter_enabled = False
        used_teleporter = False
        blocked_by_b = False
        reached_goal = False
        invalid = False
        reasons: list[str] = []

        has_blocker = discovery.observation.symbols.get("B") is not None
        has_teleporter = discovery.observation.symbols.get("T") is not None

        for step in candidate.steps:
            if step == "activate_X":
                door_open = True
            elif step == "open_D":
                if not door_open:
                    invalid = True
                    reasons.append("X toggles D before the door opens")
            elif step == "activate_A":
                teleporter_enabled = True
            elif step == "use_T":
                if not teleporter_enabled or not has_teleporter:
                    invalid = True
                    reasons.append("teleporter requires A")
                else:
                    used_teleporter = True
            elif step == "reach_G":
                if not used_teleporter and not door_open:
                    invalid = True
                    reasons.append("goal blocked without discovered rules")
                else:
                    reached_goal = True
            elif step == "move_to_goal_direct":
                invalid = True
                reasons.append("goal blocked without discovered rules")

        if has_blocker and not teleporter_enabled:
            blocked_by_b = True
        rule_accuracy = 1.0 if door_open and teleporter_enabled and used_teleporter else 0.4 if door_open or teleporter_enabled else 0.0
        plan_success = 1.0 if reached_goal and not invalid else 0.0
        rule_reuse = 1.0 if self._rule_signature(discovery.rules) in self.rule_memory else 0.0
        generalisation = 1.0 if used_teleporter and door_open and reached_goal else 0.25 if door_open else 0.0
        trial_count = float(len(candidate.steps))
        incorrect_hypotheses = 1.0 if invalid else 0.0
        score = rule_accuracy + plan_success + rule_reuse + generalisation - 0.1 * trial_count - incorrect_hypotheses
        if not reasons:
            if candidate.name == "discovered_composition":
                reasons.append("discovered interaction sequence applied to the layout")
            else:
                reasons.append("lower-value baseline strategy")
        reason = "success" if candidate.name == "discovered_composition" and plan_success > 0.0 and rule_accuracy > 0.0 else reasons[0]
        return {
            "rule_accuracy": rule_accuracy,
            "plan_success": plan_success,
            "rule_reuse": rule_reuse,
            "generalisation": generalisation,
            "trial_count": trial_count,
            "incorrect_hypotheses": incorrect_hypotheses,
            "score": score,
            "reason": reason,
            "blocked_by_B": 1.0 if blocked_by_b else 0.0,
        }

    def solve(self, raw_grid: Sequence[Sequence[str]] | Sequence[str] | Grid) -> dict[str, Any]:
        discovery = self.discovery.generate(raw_grid, context_signature=self.engine.store.context_signature({}))
        context_signature = self.engine.store.context_signature({"mode": "ctw"})
        self.engine.current_state = State(value="ctw", context={"mode": "ctw"})
        self.engine.history = [self.engine.current_state]
        candidates = self._candidate_pool(discovery, context_signature)
        self.engine.sequences = [candidate.strategy for candidate in candidates]
        selected_sequence = self.engine.select_sequence(goal={"beta": 1.0, "gamma": 2.0, "delta": 0.5, "utility": 1.0})

        best: CTWCandidate | None = None
        rejected: dict[str, str] = {}
        for candidate in candidates:
            metrics = self._simulate(candidate, discovery)
            candidate.score_trace = {key: float(value) for key, value in metrics.items() if key != "reason"}
            candidate.reason = str(metrics["reason"])
            candidate.valid = candidate.reason == "success"
            sig = self._rule_signature(discovery.rules)
            if candidate.name == "discovered_composition" and sig in self.rule_memory:
                candidate.score_trace["strategy_reuse"] = self.rule_memory[sig]
                candidate.score_trace["score"] += self.rule_memory[sig]
            else:
                candidate.score_trace["strategy_reuse"] = 0.0
            if selected_sequence is not None and tuple(selected_sequence.pattern_names) == tuple(candidate.strategy.pattern_names):
                candidate.score_trace["core_alignment"] = 1.0
                candidate.score_trace["score"] += 0.25
            else:
                candidate.score_trace["core_alignment"] = 0.0
            if not candidate.valid:
                rejected[candidate.name] = candidate.reason
            if best is None or candidate.score_trace["score"] > best.score_trace.get("score", float("-inf")):
                best = candidate

        if best is None:
            return {
                "selected_strategy": [],
                "selected_plan": [],
                "rejected": rejected,
                "rule_trace": discovery.trace,
                "reasoning_trace": {},
                "score_trace": {},
                "result": "failure",
                "reason": "no candidates",
                "confidence": 0.0,
            }

        if best.valid:
            sig = self._rule_signature(discovery.rules)
            self.rule_memory[sig] = self.rule_memory.get(sig, 0.0) + 1.0

        confidence = max(0.0, min(1.0, best.score_trace["score"] / (abs(best.score_trace["score"]) + 1.0)))
        reasoning_trace = {
            "rules": [
                {
                    "name": rule.name,
                    "antecedent": rule.antecedent,
                    "consequent": rule.consequent,
                    "rationale": rule.rationale,
                    "confidence": rule.confidence,
                }
                for rule in discovery.rules
            ],
            "generated_candidates": [
                {
                    "name": candidate.name,
                    "pattern_names": list(candidate.strategy.pattern_names),
                    "score": candidate.score_trace.get("score", 0.0),
                    "reason": candidate.reason,
                    "source": candidate.source,
                }
                for candidate in candidates
            ],
            "dependency_graph": discovery.dependency_graph,
            "selected_strategy": list(best.strategy.pattern_names),
            "score_trace": best.score_trace,
            "confidence": confidence,
            "generation_trace": discovery.trace,
        }
        return {
            "selected_strategy": list(best.strategy.pattern_names),
            "selected_plan": best.steps,
            "rejected": rejected,
            "rule_trace": discovery.trace,
            "reasoning_trace": reasoning_trace,
            "score_trace": best.score_trace,
            "result": "success" if best.valid else "failure",
            "reason": best.reason,
            "confidence": confidence,
        }
