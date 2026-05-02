"""Objective, metrics-first evaluation for the v5 stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from ..agents import AgentInput, BaseAgent
from ..arc import ArcSolver
from ..core import Pattern, PatternEngine, PatternSequence, State
from ..planning import (
    CompositionalTransformationWorldPlanner,
    DelayedConsequenceMazeBenchmark,
    LearnedUtilityBenchmark,
    NestedPrerequisiteMazePlanner,
    PrefixDisambiguationTask,
    TripleSequenceDiscoveryBenchmark,
)
from ..preprocessors.numeric import NumericPreprocessor
from ..postprocessors.numeric import NumericPostprocessor


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    name: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    benchmarks: tuple[BenchmarkScore, ...]
    overall_score: float
    passed: bool
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmarks": [
                {
                    "name": benchmark.name,
                    "score": benchmark.score,
                    "metrics": dict(benchmark.metrics),
                    "passed": benchmark.passed,
                    "trace": dict(benchmark.trace),
                }
                for benchmark in self.benchmarks
            ],
            "overall_score": self.overall_score,
            "passed": self.passed,
            "summary": dict(self.summary),
        }


class V5ObjectiveEvaluator:
    """Run a compact, numeric evaluation across the v5 stack."""

    def evaluate_core(self) -> BenchmarkScore:
        engine = PatternEngine()
        first = engine.observe(State(value=0.0, context={"mode": "objective"}))
        second = engine.observe(State(value=3.0, context={"mode": "objective"}))
        third = engine.observe(State(value=6.0, context={"mode": "objective"}))
        action = engine.act(goal={"utility": 1.0}, horizon=2)

        metrics = {
            "learned_pattern": 1.0 if first is None and second is not None and second.status == "novel" else 0.0,
            "reused_pattern": 1.0 if third is not None and third.status == "exact" else 0.0,
            "trajectory_trace": 1.0 if action.reasoning_trace is not None and action.reasoning_trace.selected_action.get("trajectory_mode") == "full" else 0.0,
            "confidence": action.confidence,
        }
        score = fmean(metrics.values())
        metrics["score"] = score
        return BenchmarkScore(
            name="core_reasoning",
            score=score,
            metrics=metrics,
            passed=score >= 0.75,
            trace={
                "selected_pattern": None if action.selected_pattern is None else action.selected_pattern.name,
                "selected_sequence": None if action.selected_sequence is None else list(action.selected_sequence.pattern_names),
                "reasoning_trace": None if action.reasoning_trace is None else action.reasoning_trace.to_dict(),
            },
        )

    def evaluate_agent(self) -> BenchmarkScore:
        agent = BaseAgent(
            name="NumericAgent",
            core=PatternEngine(),
            preprocessors=[NumericPreprocessor()],
            postprocessors=[NumericPostprocessor()],
        )
        first = agent.step(AgentInput(raw=1.0, context={"minimum": 0.0, "maximum": 10.0}))
        second = agent.step(AgentInput(raw=2.0, context={"minimum": 0.0, "maximum": 10.0}))

        metrics = {
            "trace_preserved": 1.0 if second.trace.get("adapter_trace") else 0.0,
            "valid_output": 1.0 if second.valid else 0.0,
            "confidence": second.confidence,
            "state_updated": 1.0 if agent.state.get("turn_history") else 0.0,
        }
        score = fmean(metrics.values())
        metrics["score"] = score
        return BenchmarkScore(
            name="agent_pipeline",
            score=score,
            metrics=metrics,
            passed=score >= 0.5 and first.action_type in {"unknown", "defer", "apply_delta"},
            trace={
                "first": first.trace,
                "second": second.trace,
            },
        )

    def evaluate_planning(self) -> BenchmarkScore:
        planner = NestedPrerequisiteMazePlanner()
        maze_a = [
            "S . . K1 # C",
            "# # . # # .",
            ". . . D1 . .",
            ". # # # . K2",
            ". T . . . D2",
            ". . . # # G",
        ]
        maze_b = [
            "S . K1 . # .",
            ". # . . # C",
            ". . D1 . . .",
            "# . . # K2 .",
            ". T . . D2 .",
            ". . . # # G",
        ]
        first = planner.solve(maze_a)
        second = planner.solve(maze_b)

        metrics = {
            "success": 1.0 if first.result == "success" and second.result == "success" else 0.0,
            "reuse": 1.0 if second.score_trace.get("strategy_reuse", 0.0) > 0.0 else 0.0,
            "trace": 1.0 if first.reasoning_trace.get("generation_trace") else 0.0,
        }
        score = fmean(metrics.values())
        metrics["score"] = score
        return BenchmarkScore(
            name="nested_prerequisite_planning",
            score=score,
            metrics=metrics,
            passed=score >= 0.66,
            trace={
                "first": first.reasoning_trace,
                "second": second.reasoning_trace,
            },
        )

    def evaluate_ctw(self) -> BenchmarkScore:
        planner = CompositionalTransformationWorldPlanner()
        ctw_a = [
            "S . X . D",
            "# # . # .",
            "A . T . B",
            ". . . . T",
            ". . . . G",
        ]
        ctw_b = [
            "S . . X .",
            ". # . . D",
            "A . T . .",
            ". . B . T",
            ". . . . G",
        ]
        first = planner.solve(ctw_a)
        second = planner.solve(ctw_b)

        metrics = {
            "success": 1.0 if first["result"] == "success" and second["result"] == "success" else 0.0,
            "rule_discovery": 1.0 if first["reasoning_trace"]["generation_trace"]["interactions"] else 0.0,
            "reuse": 1.0 if second["score_trace"].get("rule_reuse", 0.0) > 0.0 else 0.0,
        }
        score = fmean(metrics.values())
        metrics["score"] = score
        return BenchmarkScore(
            name="ctw_rule_discovery",
            score=score,
            metrics=metrics,
            passed=score >= 0.66,
            trace={
                "first": first["reasoning_trace"],
                "second": second["reasoning_trace"],
            },
        )

    def evaluate_dcm(self) -> BenchmarkScore:
        benchmark = DelayedConsequenceMazeBenchmark().run()
        metrics = {
            "success": 1.0 if benchmark.result == "success" else 0.0,
            "training_reward": min(1.0, max(0.0, benchmark.training_reward / 25.0)),
            "evaluation_reward": min(1.0, max(0.0, benchmark.evaluation_reward / 10.0)),
            "sequence_reuse": 1.0 if benchmark.phase_results and benchmark.phase_results[-1].sequence_length == 3 else 0.0,
        }
        score = fmean(metrics.values())
        metrics["score"] = score
        return BenchmarkScore(
            name="delayed_consequence_maze",
            score=score,
            metrics=metrics,
            passed=score >= 0.75,
            trace={
                "result": benchmark.result,
                "reason": benchmark.reason,
                "training_reward": benchmark.training_reward,
                "evaluation_reward": benchmark.evaluation_reward,
                "evaluation": benchmark.phase_results[-1].reasoning_trace if benchmark.phase_results else {},
            },
        )

    def evaluate_pdt(self) -> BenchmarkScore:
        benchmark = PrefixDisambiguationTask(use_prefix_buffer=True).run()
        metrics = {
            "success": 1.0 if benchmark.result == "success" else 0.0,
            "accuracy": benchmark.accuracy,
            "critical_accuracy": benchmark.critical_accuracy,
            "score": benchmark.critical_accuracy,
        }
        score = metrics["score"]
        return BenchmarkScore(
            name="prefix_disambiguation_task",
            score=score,
            metrics=metrics,
            passed=benchmark.result == "success",
            trace={
                "result": benchmark.result,
                "reason": benchmark.reason,
                "trace": benchmark.trace,
                "steps": [step.reasoning_trace for step in benchmark.step_results[:3]],
            },
        )

    def evaluate_lub(self) -> BenchmarkScore:
        benchmark = LearnedUtilityBenchmark().run()
        reward_score = min(1.0, max(0.0, (benchmark.average_reward + 20.0) / 30.0))
        metrics = {
            "success": 1.0 if benchmark.result == "success" else 0.0,
            "preference_accuracy": benchmark.preference_accuracy,
            "reward_score": reward_score,
            "utility_separation": benchmark.utility_separation,
        }
        score = fmean(metrics.values())
        metrics["score"] = score
        return BenchmarkScore(
            name="learned_utility_benchmark",
            score=score,
            metrics=metrics,
            passed=benchmark.result == "success",
            trace={
                "result": benchmark.result,
                "reason": benchmark.reason,
                "preference_accuracy": benchmark.preference_accuracy,
                "average_reward": benchmark.average_reward,
                "utility_separation": benchmark.utility_separation,
                "evaluation": [episode.trace for episode in benchmark.episode_results[:6]],
            },
        )

    def evaluate_tsd(self) -> BenchmarkScore:
        benchmark = TripleSequenceDiscoveryBenchmark().run()
        macro_reward = benchmark.macro_reward
        baseline_reward = benchmark.baseline_reward
        reward_gain = max(0.0, macro_reward - baseline_reward)
        metrics = {
            "success": 1.0 if benchmark.result == "success" else 0.0,
            "discovered_sequence_length": min(1.0, benchmark.discovered_sequence_length / 3.0),
            "reward_gain": min(1.0, reward_gain / 100.0),
            "macro_reward": min(1.0, max(0.0, macro_reward / 300.0)),
        }
        score = fmean(metrics.values())
        metrics["score"] = score
        return BenchmarkScore(
            name="triple_sequence_discovery",
            score=score,
            metrics=metrics,
            passed=benchmark.result == "success",
            trace={
                "result": benchmark.result,
                "reason": benchmark.reason,
                "baseline_reward": baseline_reward,
                "macro_reward": macro_reward,
                "reward_gain": reward_gain,
                "discovered_sequence_length": benchmark.discovered_sequence_length,
                "baseline": [step.reasoning_trace for step in benchmark.baseline_steps[:4]],
                "macro": [step.reasoning_trace for step in benchmark.macro_steps[:4]],
            },
        )

    def evaluate_arc(self) -> BenchmarkScore:
        solver = ArcSolver()
        tasks = [
            {
                "task_id": "toy_translate_recolour",
                "train": [
                    {"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 2], [0, 0, 0]]},
                ],
                "test": [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                "expected": [[0, 0, 0], [0, 0, 2], [0, 0, 0]],
            },
            {
                "task_id": "toy_rotate_90",
                "train": [
                    {"input": [[0, 1, 0], [0, 1, 0], [1, 1, 0]], "output": [[1, 0, 0], [1, 1, 1], [0, 0, 0]]},
                ],
                "test": [[[0, 3, 0], [0, 3, 0], [3, 3, 0]]],
                "expected": [[3, 0, 0], [3, 3, 3], [0, 0, 0]],
            },
            {
                "task_id": "toy_crop_object",
                "train": [
                    {"input": [[0, 0, 0], [0, 6, 6], [0, 6, 6]], "output": [[7, 7], [7, 7]]},
                ],
                "test": [[[0, 0, 0], [0, 6, 6], [0, 6, 6]]],
                "expected": [[7, 7], [7, 7]],
            },
        ]

        hits = 0.0
        traces: list[dict[str, Any]] = []
        for task in tasks:
            packet = solver.solve({"task_id": task["task_id"], "train": task["train"], "test": task["test"]})
            accepted = packet.context["arc"].get("accepted", False)
            correct = packet.final_output == task["expected"]
            hits += 1.0 if accepted and correct else 0.0
            traces.append(
                {
                    "task_id": task["task_id"],
                    "route": packet.context["arc"].get("route"),
                    "accepted": accepted,
                    "correct": correct,
                }
            )
        score = hits / len(tasks)
        metrics = {"task_accuracy": score, "tasks": float(len(tasks)), "score": score}
        return BenchmarkScore(
            name="arc_subset",
            score=score,
            metrics=metrics,
            passed=score >= 1.0,
            trace={"tasks": traces},
        )

    def evaluate(self) -> EvaluationReport:
        benchmarks = (
            self.evaluate_core(),
            self.evaluate_agent(),
            self.evaluate_planning(),
            self.evaluate_ctw(),
            self.evaluate_dcm(),
            self.evaluate_pdt(),
            self.evaluate_lub(),
            self.evaluate_tsd(),
            self.evaluate_arc(),
        )
        overall_score = fmean(benchmark.score for benchmark in benchmarks)
        summary = {
            "benchmark_names": [benchmark.name for benchmark in benchmarks],
            "passed_benchmarks": [benchmark.name for benchmark in benchmarks if benchmark.passed],
            "failed_benchmarks": [benchmark.name for benchmark in benchmarks if not benchmark.passed],
        }
        return EvaluationReport(
            benchmarks=benchmarks,
            overall_score=overall_score,
            passed=all(benchmark.passed for benchmark in benchmarks),
            summary=summary,
        )
