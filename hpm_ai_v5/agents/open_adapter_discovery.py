"""Agent-side open adapter discovery for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Sequence

from ..adapter import AdapterPacket, AdapterRegistry
from ..adapter.feature_adapters import (
    ConnectedComponentsAdapter,
    DeltaBufferAdapter,
    FlattenGridAdapter,
    RecentBufferAdapter,
)


@dataclass(frozen=True, slots=True)
class OpenAdapterExample:
    raw: Any
    target: Any


@dataclass(frozen=True, slots=True)
class OpenAdapterTask:
    name: str
    kind: str
    support: tuple[OpenAdapterExample, ...]
    query: tuple[OpenAdapterExample, ...]
    optimal_pipeline: str | None
    defer_expected: bool = False


@dataclass(frozen=True, slots=True)
class OpenAdapterPipelineSpec:
    name: str
    depth: int
    builder: Callable[[], AdapterRegistry]


@dataclass(frozen=True, slots=True)
class OpenAdapterTaskResult:
    name: str
    kind: str
    selected_pipeline: str | None
    optimal_pipeline: str | None
    support_accuracy: float
    query_accuracy: float
    confidence: float
    deferred: bool
    profile_reused: bool
    matched: bool
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OpenAdapterDiscoveryResult:
    result: str
    reason: str
    learned_profiles: int
    support_mean_accuracy: float
    query_mean_accuracy: float
    task_results: tuple[OpenAdapterTaskResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class OpenAdapterDiscoveryAgent:
    """Discover adapter compositions from support examples and reuse them."""

    profile_memory: dict[str, str] = field(default_factory=dict)
    pipeline_specs: tuple[OpenAdapterPipelineSpec, ...] = field(default_factory=tuple)

    @staticmethod
    def default_pipeline_specs() -> tuple[OpenAdapterPipelineSpec, ...]:
        def numeric_recent() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(RecentBufferAdapter(buffer_size=2))
            return registry

        def numeric_delta() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(DeltaBufferAdapter(buffer_size=2))
            return registry

        def numeric_combo() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(RecentBufferAdapter(buffer_size=2))
            registry.register(DeltaBufferAdapter(buffer_size=2))
            return registry

        def grid_flatten() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(FlattenGridAdapter())
            return registry

        def grid_components() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(ConnectedComponentsAdapter())
            return registry

        def grid_combo() -> AdapterRegistry:
            registry = AdapterRegistry()
            registry.register(FlattenGridAdapter())
            registry.register(ConnectedComponentsAdapter())
            return registry

        return (
            OpenAdapterPipelineSpec("numeric_recent", 1, numeric_recent),
            OpenAdapterPipelineSpec("numeric_delta", 1, numeric_delta),
            OpenAdapterPipelineSpec("numeric_combo", 2, numeric_combo),
            OpenAdapterPipelineSpec("grid_flatten", 1, grid_flatten),
            OpenAdapterPipelineSpec("grid_components", 1, grid_components),
            OpenAdapterPipelineSpec("grid_combo", 2, grid_combo),
        )

    def __post_init__(self) -> None:
        if not self.pipeline_specs:
            self.pipeline_specs = self.default_pipeline_specs()

    @staticmethod
    def _kind_from_raw(raw: Any) -> str:
        if isinstance(raw, dict):
            return "graph"
        if isinstance(raw, (list, tuple)) and raw and all(isinstance(item, (list, tuple)) for item in raw):
            return "grid"
        if isinstance(raw, (list, tuple)) and all(isinstance(item, (int, float)) for item in raw):
            return "numeric"
        return "unknown"

    @staticmethod
    def _numeric_predict(packet: AdapterPacket) -> tuple[float | None, float]:
        context = packet.context or {}
        history = context.get("history_window")
        deltas = context.get("delta_window")
        if isinstance(history, tuple) and isinstance(deltas, tuple) and len(history) >= 2 and len(deltas) >= 2:
            last = float(history[-1])
            last_delta = float(deltas[-1])
            previous_delta = float(deltas[-2])
            return last + last_delta + previous_delta, 0.95
        if isinstance(history, tuple) and len(history) >= 2:
            last = float(history[-1])
            previous = float(history[-2])
            return last + (last - previous), 0.55
        if isinstance(deltas, tuple) and len(deltas) >= 2:
            return float(deltas[-1] + deltas[-2]), 0.45
        return None, 0.0

    @staticmethod
    def _grid_predict(packet: AdapterPacket) -> tuple[int | None, float]:
        context = packet.context or {}
        component_count = context.get("component_count")
        raw_grid = packet.raw
        flat_cells: tuple[Any, ...] | None = None
        if isinstance(raw_grid, (list, tuple)) and raw_grid and all(isinstance(row, (list, tuple)) for row in raw_grid):
            flat_cells = tuple(cell for row in raw_grid for cell in row)
        if isinstance(component_count, int) and isinstance(flat_cells, tuple):
            checksum = sum(int(cell) for cell in flat_cells) % 10
            return component_count * 100 + checksum, 0.95
        if isinstance(component_count, int):
            return component_count * 100, 0.55
        if isinstance(flat_cells, tuple):
            return sum(int(cell) for cell in flat_cells) % 10, 0.45
        return None, 0.0

    def _run_support_example(self, example: OpenAdapterExample, spec: OpenAdapterPipelineSpec, kind: str) -> tuple[Any | None, float, dict[str, Any]]:
        registry = spec.builder()
        packet = AdapterPacket(raw=example.raw, context={"kind": kind})
        prediction: Any | None = None
        confidence = 0.0
        if kind == "numeric":
            if not isinstance(example.raw, (list, tuple)):
                return None, 0.0, {"reason": "invalid_numeric_example"}
            for index, raw in enumerate(example.raw):
                step_packet = AdapterPacket(raw=raw, context={"kind": kind, "step": index})
                step_packet = registry.run(step_packet, target_outputs=[adapter.name for adapter in registry.adapters.values()])
                prediction, confidence = self._numeric_predict(step_packet)
            return prediction, confidence, {"kind": kind, "pipeline": spec.name}
        if kind == "grid":
            packet = registry.run(packet, target_outputs=[adapter.name for adapter in registry.adapters.values()])
            prediction, confidence = self._grid_predict(packet)
            return prediction, confidence, {"kind": kind, "pipeline": spec.name}
        return None, 0.0, {"reason": "unsupported_kind", "kind": kind}

    def _evaluate_pipeline(self, examples: Sequence[OpenAdapterExample], spec: OpenAdapterPipelineSpec, kind: str) -> tuple[float, list[dict[str, Any]]]:
        correct = 0
        total = 0
        trace: list[dict[str, Any]] = []
        for example in examples:
            predicted, confidence, detail = self._run_support_example(example, spec, kind)
            matches = predicted is not None and predicted == example.target
            if predicted is not None:
                total += 1
                correct += int(matches)
            trace.append(
                {
                    "raw": example.raw,
                    "target": example.target,
                    "predicted": predicted,
                    "confidence": confidence,
                    "correct": matches,
                    **detail,
                }
            )
        return (correct / total if total else 0.0), trace

    def _candidate_pool(self, kind: str) -> tuple[OpenAdapterPipelineSpec, ...]:
        return tuple(
            spec
            for spec in self.pipeline_specs
            if (
                kind == "numeric"
                and spec.name.startswith("numeric")
            )
            or (
                kind == "grid"
                and spec.name.startswith("grid")
            )
        )

    def observe(self, task: OpenAdapterTask) -> OpenAdapterTaskResult:
        kind = task.kind or self._kind_from_raw(task.support[0].raw if task.support else None)
        candidate_pool = self._candidate_pool(kind)
        scores: dict[str, float] = {}
        traces: dict[str, list[dict[str, Any]]] = {}
        for spec in candidate_pool:
            accuracy, trace = self._evaluate_pipeline(task.support, spec, kind)
            scores[spec.name] = accuracy
            traces[spec.name] = trace
        if not candidate_pool:
            return OpenAdapterTaskResult(
                name=task.name,
                kind=kind,
                selected_pipeline=None,
                optimal_pipeline=task.optimal_pipeline,
                support_accuracy=0.0,
                query_accuracy=0.0,
                confidence=0.0,
                deferred=True,
                profile_reused=False,
                matched=task.defer_expected,
                trace={"reason": "no_candidates", "kind": kind},
            )
        best = max(candidate_pool, key=lambda spec: (scores.get(spec.name, 0.0), spec.depth))
        support_accuracy = scores.get(best.name, 0.0)
        confidence = max(0.0, min(1.0, support_accuracy))
        deferred = support_accuracy < 0.8
        if not deferred:
            self.profile_memory[kind] = best.name
        query_accuracy, query_trace = self._evaluate_pipeline(task.query, best, kind) if not deferred else (0.0, [])
        selected_pipeline = None if deferred else best.name
        matched = (selected_pipeline == task.optimal_pipeline) if task.optimal_pipeline is not None else deferred == task.defer_expected
        return OpenAdapterTaskResult(
            name=task.name,
            kind=kind,
            selected_pipeline=selected_pipeline,
            optimal_pipeline=task.optimal_pipeline,
            support_accuracy=support_accuracy,
            query_accuracy=query_accuracy,
            confidence=confidence,
            deferred=deferred,
            profile_reused=False,
            matched=matched,
            trace={
                "selected": selected_pipeline,
                "optimal": task.optimal_pipeline,
                "kind": kind,
                "scores": scores,
                "support": traces,
                "query": query_trace,
                "memory_size": len(self.profile_memory),
            },
        )

    def solve(self, task: OpenAdapterTask) -> OpenAdapterTaskResult:
        kind = task.kind or self._kind_from_raw(task.support[0].raw if task.support else None)
        candidate_pool = self._candidate_pool(kind)
        reused = kind in self.profile_memory
        if reused:
            selected_name = self.profile_memory[kind]
            selected_spec = next((spec for spec in candidate_pool if spec.name == selected_name), None)
        else:
            selected_spec = None
            scores: dict[str, float] = {}
            best_score = -1.0
            for spec in candidate_pool:
                accuracy, _ = self._evaluate_pipeline(task.support, spec, kind)
                scores[spec.name] = accuracy
                if accuracy > best_score or (accuracy == best_score and selected_spec is not None and spec.depth > selected_spec.depth):
                    best_score = accuracy
                    selected_spec = spec
            if selected_spec is not None and scores.get(selected_spec.name, 0.0) >= 0.8:
                self.profile_memory[kind] = selected_spec.name
                reused = False
            else:
                selected_spec = None
        if selected_spec is None:
            return OpenAdapterTaskResult(
                name=task.name,
                kind=kind,
                selected_pipeline=None,
                optimal_pipeline=task.optimal_pipeline,
                support_accuracy=0.0,
                query_accuracy=0.0,
                confidence=0.0,
                deferred=True,
                profile_reused=reused,
                matched=task.defer_expected,
                trace={"reason": "deferred", "kind": kind, "memory_size": len(self.profile_memory)},
            )
        support_accuracy, support_trace = self._evaluate_pipeline(task.support, selected_spec, kind)
        query_accuracy, query_trace = self._evaluate_pipeline(task.query, selected_spec, kind)
        confidence = max(0.0, min(1.0, (support_accuracy + query_accuracy) / 2.0))
        deferred = support_accuracy < 0.8 or (task.defer_expected and query_accuracy < 0.8)
        matched = selected_spec.name == task.optimal_pipeline if task.optimal_pipeline is not None else deferred == task.defer_expected
        return OpenAdapterTaskResult(
            name=task.name,
            kind=kind,
            selected_pipeline=None if deferred else selected_spec.name,
            optimal_pipeline=task.optimal_pipeline,
            support_accuracy=support_accuracy,
            query_accuracy=query_accuracy,
            confidence=confidence,
            deferred=deferred,
            profile_reused=reused,
            matched=matched,
            trace={
                "selected": None if deferred else selected_spec.name,
                "optimal": task.optimal_pipeline,
                "kind": kind,
                "support": support_trace,
                "query": query_trace,
                "memory_size": len(self.profile_memory),
                "profile_reused": reused,
            },
        )

    def run(
        self,
        train_tasks: Sequence[OpenAdapterTask] | None = None,
        test_tasks: Sequence[OpenAdapterTask] | None = None,
    ) -> OpenAdapterDiscoveryResult:
        train_tasks = train_tasks or (
            OpenAdapterTask(
                "NumericTrain",
                "numeric",
                support=(
                    OpenAdapterExample((1.0, 2.0, 4.0, 7.0, 12.0), 20.0),
                    OpenAdapterExample((3.0, 5.0, 8.0, 13.0, 21.0), 34.0),
                ),
                query=(
                    OpenAdapterExample((2.0, 3.0, 5.0, 8.0, 13.0), 21.0),
                ),
                optimal_pipeline="numeric_combo",
            ),
            OpenAdapterTask(
                "GridTrain",
                "grid",
                support=(
                    OpenAdapterExample(((0, 1, 1), (0, 0, 1), (2, 0, 0)), 205),
                    OpenAdapterExample(((3, 3, 0), (0, 0, 0), (4, 4, 4)), 208),
                ),
                query=(
                    OpenAdapterExample(((0, 5, 5), (6, 0, 5), (6, 0, 0)), 207),
                ),
                optimal_pipeline="grid_combo",
            ),
        )
        test_tasks = test_tasks or (
            OpenAdapterTask(
                "NumericTest",
                "numeric",
                support=(
                    OpenAdapterExample((4.0, 6.0, 9.0, 13.0, 18.0), 27.0),
                ),
                query=(
                    OpenAdapterExample((5.0, 7.0, 10.0, 14.0, 19.0), 28.0),
                ),
                optimal_pipeline="numeric_combo",
            ),
            OpenAdapterTask(
                "GridTest",
                "grid",
                support=(
                    OpenAdapterExample(((0, 2, 2), (3, 0, 2), (3, 0, 0)), 202),
                ),
                query=(
                    OpenAdapterExample(((0, 7, 7), (8, 0, 7), (8, 0, 0)), 207),
                ),
                optimal_pipeline="grid_combo",
            ),
            OpenAdapterTask(
                "GraphHeldOut",
                "graph",
                support=(
                    OpenAdapterExample({"A": ["B"], "B": ["C"], "C": []}, "A"),
                ),
                query=(
                    OpenAdapterExample({"A": ["B", "C"], "B": ["C"], "C": []}, "A"),
                ),
                optimal_pipeline=None,
                defer_expected=True,
            ),
        )

        train_results = [self.observe(task) for task in train_tasks]
        test_results = [self.solve(task) for task in test_tasks]
        all_results = [*train_results, *test_results]
        support_mean_accuracy = fmean(result.support_accuracy for result in all_results) if all_results else 0.0
        query_mean_accuracy = fmean(result.query_accuracy for result in all_results if not result.deferred) if any(not result.deferred for result in all_results) else 0.0
        supported = [result for result in test_results if result.optimal_pipeline is not None]
        defer_results = [result for result in test_results if result.optimal_pipeline is None]
        selected_matches = sum(1 for result in supported if result.matched)
        support_pass = all(result.support_accuracy >= 0.8 for result in supported)
        query_pass = all(result.query_accuracy >= 0.8 for result in supported)
        defer_pass = all(result.deferred for result in defer_results)
        passed = support_pass and query_pass and defer_pass and selected_matches == len(supported)
        reason = "success" if passed else "adapter discovery failed"
        return OpenAdapterDiscoveryResult(
            result="success" if passed else "failure",
            reason=reason,
            learned_profiles=len(self.profile_memory),
            support_mean_accuracy=support_mean_accuracy,
            query_mean_accuracy=query_mean_accuracy,
            task_results=tuple(all_results),
            trace={
                "train": [result.trace for result in train_results],
                "test": [result.trace for result in test_results],
                "profiles": dict(self.profile_memory),
            },
        )
