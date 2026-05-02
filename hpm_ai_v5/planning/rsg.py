"""Rotating Sequence Generalization benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core import CoreConfig, PatternEngine, State


@dataclass(frozen=True, slots=True)
class RSGPhaseResult:
    phase: int
    sequence_length: int
    accuracy: float
    predictions: list[dict[str, Any]]
    reasoning_trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RSGResult:
    base_block: tuple[float, ...]
    phases: tuple[int, ...]
    horizon: int
    phase_results: tuple[RSGPhaseResult, ...]
    result: str
    reason: str
    trace: dict[str, Any]


@dataclass(slots=True)
class RotatingSequenceGeneralizationBenchmark:
    """Evaluate phase-invariant periodic sequence learning."""

    config: CoreConfig = field(default_factory=lambda: CoreConfig(near_threshold=0.0, history_limit=32))

    def _rotated_block(self, block: Sequence[float], phase: int) -> tuple[float, ...]:
        values = tuple(float(item) for item in block)
        if not values:
            return values
        phase = phase % len(values)
        return values[phase:] + values[:phase]

    def _build_stream(self, block: Sequence[float], phase: int, *, cycles: int = 4) -> list[float]:
        rotated = self._rotated_block(block, phase)
        deltas = list(rotated) * max(1, cycles)
        values = [0.0]
        for delta in deltas:
            values.append(values[-1] + float(delta))
        return values

    def _evaluate_phase(self, block: Sequence[float], phase: int, horizon: int, cycles: int) -> RSGPhaseResult:
        engine = PatternEngine(config=self.config)
        stream = self._build_stream(block, phase, cycles=cycles)
        warmup = 2 * len(block) + 1
        predictions: list[dict[str, Any]] = []

        for index, token in enumerate(stream):
            state = State(value=token, context={"phase": phase, "position": index, "mode": "rsg"})
            if index < warmup:
                engine.observe(state)
                continue
            target_index = index + horizon
            if index < warmup or target_index >= len(stream):
                continue
            engine.current_state = state
            if not engine.history or engine.history[-1].value != state.value:
                engine.history = (engine.history + [state])[-engine.config.history_limit :]
            action = engine.act(goal={"utility": 1.0, "plan_horizon": float(horizon)}, horizon=horizon)
            predicted = action.forecast.value if action.action_type == "apply_delta" and action.forecast is not None else None
            actual = stream[target_index]
            predictions.append(
                {
                    "index": index,
                    "target_index": target_index,
                    "predicted": predicted,
                    "actual": actual,
                    "correct": predicted == actual,
                    "selected_sequence": None if action.selected_sequence is None else list(action.selected_sequence.pattern_names),
                    "trajectory_mode": action.trace.get("trajectory_mode"),
                    "confidence": action.confidence,
                }
            )

        selected_sequence = engine.select_sequence(goal={"utility": 1.0, "plan_horizon": float(horizon)})
        discovered_length = 0 if selected_sequence is None else len(selected_sequence.pattern_names)
        correct = sum(1 for item in predictions if item["correct"])
        accuracy = correct / len(predictions) if predictions else 0.0
        return RSGPhaseResult(
            phase=phase,
            sequence_length=discovered_length,
            accuracy=accuracy,
            predictions=predictions,
            reasoning_trace={
                "selected_sequence": None if selected_sequence is None else list(selected_sequence.pattern_names),
                "sequence_support": None if selected_sequence is None else selected_sequence.support,
                "sequence_density": None if selected_sequence is None else selected_sequence.density,
                "pattern_trace": list(engine.pattern_trace),
            },
        )

    def evaluate(self, base_block: Sequence[float], phases: Sequence[int], *, horizon: int | None = None, cycles: int = 4) -> RSGResult:
        block = tuple(float(item) for item in base_block)
        if not block:
            raise ValueError("base_block must not be empty")
        horizon = len(block) if horizon is None else horizon
        cycles = max(cycles, 4)
        phase_results = tuple(self._evaluate_phase(block, phase, horizon, cycles) for phase in phases)
        passed = all(result.sequence_length == len(block) and result.accuracy >= 0.9 for result in phase_results)
        reason = "success" if passed else "sequence discovery or prediction failed"
        trace = {
            "base_block": list(block),
            "phases": list(phases),
            "horizon": horizon,
            "phase_results": [
                {
                    "phase": result.phase,
                    "sequence_length": result.sequence_length,
                    "accuracy": result.accuracy,
                    "predictions": result.predictions,
                }
                for result in phase_results
            ],
        }
        return RSGResult(
            base_block=block,
            phases=tuple(int(phase) for phase in phases),
            horizon=horizon,
            phase_results=phase_results,
            result="success" if passed else "failure",
            reason=reason,
            trace=trace,
        )
