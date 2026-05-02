"""Polygraph Agreement Benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Sequence

from ..core import CoreConfig, PatternEngine, State
from ..core.evaluator import PolygraphAgreement, PolygraphEvaluator, PolygraphScore
from ..polygraphs.base import PolygraphView


@dataclass(frozen=True, slots=True)
class PABStepResult:
    step: int
    raw: float
    actual_next: float | None
    exact_prediction: float | None
    noisy_prediction: float | None
    trend_prediction: float | None
    selected_view: str | None
    agreement_key: str | None
    agreement_score: float
    view_scores: dict[str, float]
    reasoning_trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PABResult:
    result: str
    reason: str
    agreement_accuracy: float
    noisy_accuracy: float
    exact_accuracy: float
    consensus_rate: float
    view_gap: float
    steps: tuple[PABStepResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class TriangleWavePolygraphGenerator:
    """Generate exact, noisy, and trend views for a triangle wave."""

    previous_value: float | None = None
    step: int = 0

    def _noise(self, value: float) -> float:
        cycle = self.step % 10
        if cycle in {2, 6, 9}:
            return value + (8.0 if cycle == 2 else -7.0 if cycle == 6 else 5.0)
        return value

    def _trend(self, value: float) -> float:
        if self.previous_value is None:
            return 0.0
        delta = value - self.previous_value
        if delta > 0.0:
            return 1.0
        if delta < 0.0:
            return -1.0
        return 0.0

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        value = float(raw)
        context = dict(context or {})
        context.setdefault("mode", "pab")
        context.setdefault("domain", "numeric")
        trend_value = self._trend(value)
        noisy_value = self._noise(value)
        views = [
            PolygraphView(
                name="exact",
                state=State(value=value, context={**context, "view": "exact"}),
                context={**context, "view": "exact"},
            ),
            PolygraphView(
                name="noisy",
                state=State(value=noisy_value, context={**context, "view": "noisy"}),
                context={**context, "view": "noisy"},
            ),
            PolygraphView(
                name="trend",
                state=State(value=value, context={**context, "view": "trend", "trend": trend_value}),
                context={**context, "view": "trend", "trend": trend_value},
            ),
        ]
        self.previous_value = value
        self.step += 1
        return views


@dataclass(slots=True)
class PolygraphAgreementAgent:
    """Maintain one engine per view and fuse them with agreement scoring."""

    config: CoreConfig = field(default_factory=lambda: CoreConfig(near_threshold=0.0, history_limit=32))
    evaluator: PolygraphEvaluator = field(default_factory=PolygraphEvaluator)
    engines: dict[str, PatternEngine] = field(default_factory=dict)

    def _action_prediction(self, action) -> float | None:
        if action.action_type != "apply_delta" or action.forecast is None:
            return None
        if not isinstance(action.forecast.value, (int, float)):
            return None
        return float(action.forecast.value)

    def step(
        self,
        raw: float,
        *,
        context: dict[str, Any] | None = None,
        generator: TriangleWavePolygraphGenerator | None = None,
        goal: dict[str, float] | None = None,
    ) -> tuple[PABStepResult, dict[str, float]]:
        generator = generator or TriangleWavePolygraphGenerator()
        views = generator.generate(raw, context=context)
        goal = goal or {"utility": 1.0}
        view_actions: dict[str, Any] = {}
        view_scores: dict[str, PolygraphScore] = {}

        for view in views:
            engine = self.engines.setdefault(view.name, PatternEngine(config=self.config))
            engine.observe(view.state)
            view_scores[view.name] = self.evaluator.score_engine(engine)
            horizon = int(goal.get("plan_horizon", 3.0)) if goal else 3
            view_actions[view.name] = engine.act(goal=goal, horizon=max(1, horizon))

        agreement = self.evaluator.agreement(view_actions, view_scores)
        selected_view = self.evaluator.select_view(view_scores)
        if agreement.selected_key is not None:
            for view_name, action in view_actions.items():
                if self.evaluator._action_key(action) == agreement.selected_key:
                    selected_view = view_name
                    break

        selected_action = view_actions.get(selected_view) if selected_view is not None else None
        exact_action = view_actions.get("exact")
        noisy_action = view_actions.get("noisy")
        trend_action = view_actions.get("trend")
        exact_prediction = self._action_prediction(exact_action) if exact_action is not None else None
        noisy_prediction = self._action_prediction(noisy_action) if noisy_action is not None else None
        trend_prediction = self._action_prediction(trend_action) if trend_action is not None else None

        step_result = PABStepResult(
            step=int(context.get("step", 0) if context else 0),
            raw=raw,
            actual_next=float(context["next_value"]) if context and isinstance(context.get("next_value"), (int, float)) else None,
            exact_prediction=exact_prediction,
            noisy_prediction=noisy_prediction,
            trend_prediction=trend_prediction,
            selected_view=selected_view,
            agreement_key=agreement.selected_key,
            agreement_score=agreement.score,
            view_scores={name: score.score for name, score in view_scores.items()},
            reasoning_trace={
                "agreement": {
                    "selected_key": agreement.selected_key,
                    "support": agreement.support,
                    "dispersion": agreement.dispersion,
                    "score": agreement.score,
                },
                "selected_view": selected_view,
                "selected_action": None if selected_action is None else selected_action.reasoning_trace.to_dict() if selected_action.reasoning_trace is not None else {},
                "view_scores": {name: score.score for name, score in view_scores.items()},
            },
        )
        return step_result, {name: score.score for name, score in view_scores.items()}


@dataclass(slots=True)
class PolygraphAgreementBenchmark:
    """Compare noisy-only prediction against agreement across exact, noisy, and trend views."""

    config: CoreConfig = field(default_factory=lambda: CoreConfig(near_threshold=0.0, history_limit=32))
    engine: PolygraphAgreementAgent = field(init=False)
    generator: TriangleWavePolygraphGenerator = field(default_factory=TriangleWavePolygraphGenerator)
    stream: Sequence[float] = field(default_factory=lambda: (1.0, 2.0))
    cycles: int = 24

    def __post_init__(self) -> None:
        self.engine = PolygraphAgreementAgent(config=self.config)

    def _build_stream(self) -> list[float]:
        values = list(self.stream)
        if not values:
            raise ValueError("stream must not be empty")
        return values * max(1, self.cycles)

    def run(self) -> PABResult:
        stream = self._build_stream()
        warmup = 2 * len(self.stream)
        steps: list[PABStepResult] = []
        agreement_choices: list[bool] = []
        exact_choices: list[bool] = []
        noisy_choices: list[bool] = []
        selected_views: list[str] = []

        for index, raw in enumerate(stream[:-1]):
            context = {"step": index, "next_value": stream[index + 1], "mode": "pab"}
            step_result, _ = self.engine.step(raw, context=context, generator=self.generator, goal={"utility": 1.0, "min_confidence": 0.0, "plan_horizon": 1.0})
            steps.append(step_result)
            if index < warmup:
                continue
            selected_views.append(step_result.selected_view or "")
            agreement_choices.append(step_result.selected_view in {"exact", "trend"})
            exact_choices.append(step_result.selected_view == "exact")
            noisy_choices.append(step_result.selected_view == "noisy")

        agreement_accuracy = sum(1 for item in agreement_choices if item) / len(agreement_choices) if agreement_choices else 0.0
        exact_accuracy = sum(1 for item in exact_choices if item) / len(exact_choices) if exact_choices else 0.0
        noisy_accuracy = sum(1 for item in noisy_choices if item) / len(noisy_choices) if noisy_choices else 0.0
        consensus_rate = sum(1 for view in selected_views if view in {"exact", "trend"}) / len(selected_views) if selected_views else 0.0
        view_scores = [step.view_scores for step in steps]
        exact_view_mean = fmean(score_map.get("exact", 0.0) for score_map in view_scores) if view_scores else 0.0
        trend_view_mean = fmean(score_map.get("trend", 0.0) for score_map in view_scores) if view_scores else 0.0
        noisy_view_mean = fmean(score_map.get("noisy", 0.0) for score_map in view_scores) if view_scores else 0.0
        view_gap = max(0.0, ((exact_view_mean + trend_view_mean) / 2.0) - noisy_view_mean)
        normalized_gap = max(0.0, min(1.0, view_gap / (abs(exact_view_mean) + abs(trend_view_mean) + 1.0)))
        score = fmean([agreement_accuracy, exact_accuracy, consensus_rate, normalized_gap])
        passed = agreement_accuracy >= 0.95 and agreement_accuracy >= noisy_accuracy + 0.15 and consensus_rate >= 0.7
        reason = "success" if passed else "polygraph agreement failed"
        trace = {
            "agreement_accuracy": agreement_accuracy,
            "exact_accuracy": exact_accuracy,
            "noisy_accuracy": noisy_accuracy,
            "consensus_rate": consensus_rate,
            "view_gap": view_gap,
            "steps": [step.reasoning_trace for step in steps[:6]],
        }
        return PABResult(
            result="success" if passed else "failure",
            reason=reason,
            agreement_accuracy=agreement_accuracy,
            noisy_accuracy=noisy_accuracy,
            exact_accuracy=exact_accuracy,
            consensus_rate=consensus_rate,
            view_gap=view_gap,
            steps=tuple(steps),
            trace={**trace, "score": score},
        )
