"""Automatic adapter composition for v5 benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Sequence

from ..adapter import AdapterPacket, AdapterRegistry
from ..core import PatternEngine
from ..adapter.feature_adapters import NumericAdapter, PrefixBufferAdapter, StateFusionAdapter


@dataclass(frozen=True, slots=True)
class AACTask:
    name: str
    family: str
    stream: tuple[float, ...]
    optimal_pipeline: str
    focus_indices: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class AACPipelineSpec:
    name: str
    terminal: str
    depth: int
    builder: Callable[[], AdapterRegistry]


@dataclass(frozen=True, slots=True)
class AACTaskResult:
    name: str
    family: str
    selected_pipeline: str
    optimal_pipeline: str
    profile: tuple[float, ...]
    calibration_accuracy: float
    evaluation_accuracy: float
    matched: bool
    profile_reused: bool
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AACResult:
    result: str
    reason: str
    learned_profiles: int
    training_mean_accuracy: float
    test_mean_accuracy: float
    task_results: tuple[AACTaskResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class AutomaticAdapterComposer:
    """Learn which adapter pipeline best fits a structural task signature."""

    pipeline_specs: tuple[AACPipelineSpec, ...] = field(default_factory=tuple)
    profile_memory: dict[tuple[float, ...], str] = field(default_factory=dict)

    @staticmethod
    def default_pipeline_specs() -> tuple[AACPipelineSpec, ...]:
        def numeric_registry() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(NumericAdapter())
            return registry

        def prefix_registry() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(PrefixBufferAdapter(buffer_size=2, value_mode="tuple", include_history_context=True))
            return registry

        def fused_registry() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(NumericAdapter())
            registry.register(PrefixBufferAdapter(buffer_size=3, value_mode="tuple", include_history_context=True))
            registry.register(StateFusionAdapter())
            return registry

        return (
            AACPipelineSpec(name="numeric", terminal="numeric", depth=1, builder=numeric_registry),
            AACPipelineSpec(name="prefix_buffer", terminal="prefix_buffer", depth=1, builder=prefix_registry),
            AACPipelineSpec(name="state_fusion", terminal="state_fusion", depth=3, builder=fused_registry),
        )

    def __post_init__(self) -> None:
        if not self.pipeline_specs:
            self.pipeline_specs = self.default_pipeline_specs()

    def _run_pipeline(
        self,
        stream: Sequence[float],
        spec: AACPipelineSpec,
        *,
        focus_indices: set[int] | None = None,
    ) -> tuple[float, float, tuple[float, ...], dict[str, Any]]:
        registry = spec.builder()
        engine = PatternEngine()
        calibration_correct = 0
        calibration_total = 0
        evaluation_correct = 0
        evaluation_total = 0
        predictions: list[dict[str, Any]] = []
        previous_prediction: Any = None
        previous_history_window: Any = None
        terminal_adapter = registry.adapters.get(spec.terminal)
        calibration_limit = max(4, min(len(stream) - 1, 5))
        if len(stream) < 2:
            return 0.0, 0.0, tuple(), {"reason": "stream_too_short"}

        for index, raw in enumerate(stream):
            packet = AdapterPacket(raw=raw, context={"domain": "aac", "step": index})
            packet = registry.run(packet, target_outputs=[spec.terminal])
            state = packet.states[-1]
            history_window = state.context.get("history_window")
            if previous_prediction is not None:
                actual = state.value
                correct = previous_prediction == actual
                scored = focus_indices is None or index in focus_indices
                bucket = "calibration" if index <= calibration_limit else "evaluation"
                if scored:
                    if bucket == "calibration":
                        calibration_total += 1
                        if correct:
                            calibration_correct += 1
                    else:
                        evaluation_total += 1
                        if correct:
                            evaluation_correct += 1
                predictions.append(
                    {
                        "index": index,
                        "actual": actual,
                        "predicted": previous_prediction,
                        "correct": correct,
                        "bucket": bucket,
                        "scored": scored,
                    }
                )
            if (
                terminal_adapter is not None
                and hasattr(terminal_adapter, "record_transition")
                and previous_history_window is not None
            ):
                terminal_adapter.record_transition(previous_history_window, state.value)
            engine.observe(state)
            predicted: Any | None = None
            if terminal_adapter is not None and hasattr(terminal_adapter, "predict_transition") and history_window is not None:
                predicted = terminal_adapter.predict_transition(history_window)
            action = engine.act(goal={"utility": 0.0, "alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25, "min_confidence": 0.3}, horizon=1)
            if predicted is None:
                predicted = None if action.forecast is None else action.forecast.value
            previous_prediction = predicted
            previous_history_window = history_window

        calibration_accuracy = calibration_correct / calibration_total if calibration_total else 0.0
        evaluation_accuracy = evaluation_correct / evaluation_total if evaluation_total else 0.0
        profile = (round(calibration_accuracy, 3), round(evaluation_accuracy, 3))
        trace = {
            "pipeline": spec.name,
            "calibration_accuracy": calibration_accuracy,
            "evaluation_accuracy": evaluation_accuracy,
            "predictions": predictions[:6],
        }
        return calibration_accuracy, evaluation_accuracy, profile, trace

    def observe(
        self,
        name: str,
        family: str,
        stream: Sequence[float],
        optimal_pipeline: str,
        *,
        focus_indices: set[int] | None = None,
    ) -> AACTaskResult:
        scores: dict[str, float] = {}
        traces: dict[str, dict[str, Any]] = {}
        evaluation: dict[str, float] = {}
        for spec in self.pipeline_specs:
            calibration_accuracy, evaluation_accuracy, profile, trace = self._run_pipeline(
                stream,
                spec,
                focus_indices=focus_indices,
            )
            scores[spec.name] = calibration_accuracy
            evaluation[spec.name] = evaluation_accuracy
            traces[spec.name] = {**trace, "profile": profile}

        profile = tuple(scores[spec.name] for spec in self.pipeline_specs)
        selected_spec = max(self.pipeline_specs, key=lambda spec: (evaluation[spec.name], scores[spec.name], spec.depth))
        self.profile_memory[profile] = optimal_pipeline
        matched = selected_spec.name == optimal_pipeline
        return AACTaskResult(
            name=name,
            family=family,
            selected_pipeline=selected_spec.name,
            optimal_pipeline=optimal_pipeline,
            profile=profile,
            calibration_accuracy=scores[selected_spec.name],
            evaluation_accuracy=evaluation[selected_spec.name],
            matched=matched,
            profile_reused=False,
            trace={
                "scores": scores,
                "evaluation": evaluation,
                "selected": selected_spec.name,
                "optimal": optimal_pipeline,
                "memory_size": len(self.profile_memory),
            },
        )

    def solve(
        self,
        name: str,
        family: str,
        stream: Sequence[float],
        optimal_pipeline: str,
        *,
        focus_indices: set[int] | None = None,
    ) -> AACTaskResult:
        scores: dict[str, float] = {}
        evaluation: dict[str, float] = {}
        traces: dict[str, dict[str, Any]] = {}
        for spec in self.pipeline_specs:
            calibration_accuracy, evaluation_accuracy, profile, trace = self._run_pipeline(
                stream,
                spec,
                focus_indices=focus_indices,
            )
            scores[spec.name] = calibration_accuracy
            evaluation[spec.name] = evaluation_accuracy
            traces[spec.name] = {**trace, "profile": profile}

        profile = tuple(scores[spec.name] for spec in self.pipeline_specs)
        selected_name = self.profile_memory.get(profile)
        profile_reused = selected_name is not None
        if selected_name is None:
            selected_spec = max(self.pipeline_specs, key=lambda spec: (evaluation[spec.name], scores[spec.name], spec.depth))
            selected_name = selected_spec.name
        selected_spec = next(spec for spec in self.pipeline_specs if spec.name == selected_name)
        matched = selected_name == optimal_pipeline
        return AACTaskResult(
            name=name,
            family=family,
            selected_pipeline=selected_name,
            optimal_pipeline=optimal_pipeline,
            profile=profile,
            calibration_accuracy=scores[selected_name],
            evaluation_accuracy=evaluation[selected_name],
            matched=matched,
            profile_reused=profile_reused,
            trace={
                "scores": scores,
                "evaluation": evaluation,
                "selected": selected_name,
                "optimal": optimal_pipeline,
                "profile_reused": profile_reused,
                "memory_size": len(self.profile_memory),
                "selected_depth": selected_spec.depth,
            },
        )

    def run(
        self,
        train_tasks: Sequence[AACTask] | None = None,
        test_tasks: Sequence[AACTask] | None = None,
    ) -> AACResult:
        train_tasks = train_tasks or (
            AACTask("ScalarTrain", "numeric", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), "numeric"),
            AACTask("PrefixTrain", "prefix", (1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0), "prefix_buffer"),
            AACTask("FusionTrain", "fusion", (1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0), "state_fusion"),
        )
        test_tasks = test_tasks or (
            AACTask("ScalarTest", "numeric", (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0), "numeric"),
            AACTask("PrefixTest", "prefix", (3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0), "prefix_buffer"),
            AACTask("FusionTest", "fusion", (5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0), "state_fusion"),
        )

        train_results = [
            self.observe(task.name, task.family, task.stream, task.optimal_pipeline, focus_indices=task.focus_indices)
            for task in train_tasks
        ]
        test_results = [
            self.solve(task.name, task.family, task.stream, task.optimal_pipeline, focus_indices=task.focus_indices)
            for task in test_tasks
        ]
        all_results = [*train_results, *test_results]
        training_mean_accuracy = fmean(result.evaluation_accuracy for result in train_results) if train_results else 0.0
        test_mean_accuracy = fmean(result.evaluation_accuracy for result in test_results) if test_results else 0.0
        selected_matches = sum(1 for result in test_results if result.matched)
        reused_profiles = sum(1 for result in test_results if result.profile_reused)
        passed = test_mean_accuracy >= 0.8 and selected_matches == len(test_results) and reused_profiles >= 1
        reason = "success" if passed else "adapter composition failed"
        return AACResult(
            result="success" if passed else "failure",
            reason=reason,
            learned_profiles=len(self.profile_memory),
            training_mean_accuracy=training_mean_accuracy,
            test_mean_accuracy=test_mean_accuracy,
            task_results=tuple(all_results),
            trace={
                "train": [result.trace for result in train_results],
                "test": [result.trace for result in test_results],
                "profiles": [
                    {
                        "task": result.name,
                        "family": result.family,
                        "selected_pipeline": result.selected_pipeline,
                        "optimal_pipeline": result.optimal_pipeline,
                        "profile": list(result.profile),
                    }
                    for result in all_results
                ],
            },
        )


@dataclass(slots=True)
class AutomaticAdapterCompositionBenchmark:
    """Learn adapter pipelines from training tasks and reuse them on variants."""

    composer: AutomaticAdapterComposer = field(default_factory=AutomaticAdapterComposer)
    train_tasks: tuple[tuple[str, str, tuple[float, ...], str], ...] = field(default_factory=lambda: (
        ("ScalarTrain", "numeric", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), "numeric"),
        ("PrefixTrain", "prefix", (1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0), "prefix_buffer"),
        ("FusionTrain", "fusion", (1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0), "state_fusion"),
    ))
    test_tasks: tuple[tuple[str, str, tuple[float, ...], str], ...] = field(default_factory=lambda: (
        ("ScalarTest", "numeric", (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0), "numeric"),
        ("PrefixTest", "prefix", (3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0), "prefix_buffer"),
        ("FusionTest", "fusion", (5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0), "state_fusion"),
    ))

    def run(self) -> AACResult:
        train_results = [
            self.composer.observe(name, family, stream, optimal_pipeline)
            for name, family, stream, optimal_pipeline in self.train_tasks
        ]
        test_results = [
            self.composer.solve(name, family, stream, optimal_pipeline)
            for name, family, stream, optimal_pipeline in self.test_tasks
        ]
        all_results = [*train_results, *test_results]
        training_mean_accuracy = fmean(result.evaluation_accuracy for result in train_results) if train_results else 0.0
        test_mean_accuracy = fmean(result.evaluation_accuracy for result in test_results) if test_results else 0.0
        selected_matches = sum(1 for result in test_results if result.matched)
        reused_profiles = sum(1 for result in test_results if result.profile_reused)
        passed = test_mean_accuracy >= 0.8 and selected_matches == len(test_results) and reused_profiles >= 1
        reason = "success" if passed else "adapter composition failed"
        return AACResult(
            result="success" if passed else "failure",
            reason=reason,
            learned_profiles=len(self.composer.profile_memory),
            training_mean_accuracy=training_mean_accuracy,
            test_mean_accuracy=test_mean_accuracy,
            task_results=tuple(all_results),
            trace={
                "train": [result.trace for result in train_results],
                "test": [result.trace for result in test_results],
                "profiles": [
                    {
                        "task": result.name,
                        "family": result.family,
                        "selected_pipeline": result.selected_pipeline,
                        "optimal_pipeline": result.optimal_pipeline,
                        "profile": list(result.profile),
                    }
                    for result in all_results
                ],
            },
        )
