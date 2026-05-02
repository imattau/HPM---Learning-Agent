"""Automatic Adapter Composition benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from ..agents import AutomaticAdapterComposer


@dataclass(frozen=True, slots=True)
class AACBenchmarkTaskResult:
    name: str
    family: str
    selected_pipeline: str
    optimal_pipeline: str
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
    task_results: tuple[AACBenchmarkTaskResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class AutomaticAdapterCompositionBenchmark:
    """Learn preprocessing compositions and reuse them on held-out variants."""

    composer: AutomaticAdapterComposer = field(default_factory=AutomaticAdapterComposer)

    def run(self) -> AACResult:
        task_results = self.composer.run()
        benchmark_results = tuple(
            AACBenchmarkTaskResult(
                name=result.name,
                family=result.family,
                selected_pipeline=result.selected_pipeline,
                optimal_pipeline=result.optimal_pipeline,
                calibration_accuracy=result.calibration_accuracy,
                evaluation_accuracy=result.evaluation_accuracy,
                matched=result.matched,
                profile_reused=result.profile_reused,
                trace=result.trace,
            )
            for result in task_results.task_results
        )
        training_results = benchmark_results[:3]
        test_results = benchmark_results[3:]
        training_mean_accuracy = fmean(result.evaluation_accuracy for result in training_results) if training_results else 0.0
        test_mean_accuracy = fmean(result.evaluation_accuracy for result in test_results) if test_results else 0.0
        return AACResult(
            result=task_results.result,
            reason=task_results.reason,
            learned_profiles=task_results.learned_profiles,
            training_mean_accuracy=training_mean_accuracy,
            test_mean_accuracy=test_mean_accuracy,
            task_results=benchmark_results,
            trace=task_results.trace,
        )
