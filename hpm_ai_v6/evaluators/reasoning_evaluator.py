from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ReasoningBenchmarkCase:
    name: str
    question: str
    method: str = "auto"
    expected_intent: Optional[str] = None
    expected_mode: Optional[str] = None
    expected_answer_contains: Sequence[str] = ()
    forbidden_answer_contains: Sequence[str] = ()
    expected_nodes: Sequence[str] = ()
    expected_relations: Sequence[str] = ()
    require_chosen_path: bool = False
    max_steps: Optional[int] = None
    notes: str = ""


@dataclass
class ReasoningCaseResult:
    case: ReasoningBenchmarkCase
    success: bool
    latency_ms: float
    proof_steps: int
    trace: Dict[str, Any]
    failures: List[str] = field(default_factory=list)


@dataclass
class ReasoningBenchmarkReport:
    case_results: List[ReasoningCaseResult]

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for result in self.case_results if result.success)

    @property
    def accuracy(self) -> float:
        if not self.case_results:
            return 0.0
        return self.passed_cases / len(self.case_results)

    @property
    def average_latency_ms(self) -> float:
        if not self.case_results:
            return 0.0
        return mean(result.latency_ms for result in self.case_results)

    @property
    def average_proof_steps(self) -> float:
        if not self.case_results:
            return 0.0
        return mean(result.proof_steps for result in self.case_results)

    def by_method(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, List[ReasoningCaseResult]] = {}
        for result in self.case_results:
            summary.setdefault(result.case.method, []).append(result)
        return {
            method: {
                "accuracy": sum(1 for result in results if result.success) / len(results),
                "avg_latency_ms": mean(result.latency_ms for result in results),
                "avg_proof_steps": mean(result.proof_steps for result in results),
                "count": float(len(results)),
            }
            for method, results in summary.items()
        }

    def render_text(self) -> str:
        lines = [
            f"cases={self.total_cases}",
            f"passed={self.passed_cases}",
            f"accuracy={self.accuracy:.3f}",
            f"avg_latency_ms={self.average_latency_ms:.2f}",
            f"avg_proof_steps={self.average_proof_steps:.2f}",
        ]
        for method, stats in sorted(self.by_method().items()):
            lines.append(
                f"{method}: accuracy={stats['accuracy']:.3f}, avg_latency_ms={stats['avg_latency_ms']:.2f}, "
                f"avg_proof_steps={stats['avg_proof_steps']:.2f}, count={int(stats['count'])}"
            )
        for result in self.case_results:
            status = "PASS" if result.success else "FAIL"
            suffix = "" if result.success else f" failures={'; '.join(result.failures)}"
            lines.append(
                f"{status} {result.case.name} [{result.case.method}] latency={result.latency_ms:.2f}ms "
                f"steps={result.proof_steps}{suffix}"
            )
        return "\n".join(lines)


class ReasoningEvaluator:
    def __init__(self, reasoning_agent: Any):
        self.reasoning_agent = reasoning_agent

    def evaluate_case(self, case: ReasoningBenchmarkCase) -> ReasoningCaseResult:
        start = time.perf_counter()
        trace = self.reasoning_agent.reason_with_trace(case.question, method=case.method)
        latency_ms = (time.perf_counter() - start) * 1000.0

        failures: List[str] = []
        chosen_path = trace.get("chosen_path") or {}
        steps = list(chosen_path.get("steps") or [])
        nodes = [node.get("label", "") for node in (chosen_path.get("nodes") or [])]
        relations = [step.get("relation", "") for step in steps]
        answer = str(trace.get("answer", ""))

        if case.expected_intent and trace.get("intent") != case.expected_intent:
            failures.append(f"intent={trace.get('intent')} expected={case.expected_intent}")
        if case.expected_mode and trace.get("mode") != case.expected_mode:
            failures.append(f"mode={trace.get('mode')} expected={case.expected_mode}")
        for snippet in case.expected_answer_contains:
            if snippet.lower() not in answer.lower():
                failures.append(f"missing answer snippet={snippet!r}")
        for snippet in case.forbidden_answer_contains:
            if snippet.lower() in answer.lower():
                failures.append(f"forbidden answer snippet={snippet!r}")
        if case.require_chosen_path and not trace.get("chosen_path"):
            failures.append("chosen_path missing")
        if case.expected_nodes and list(case.expected_nodes) != nodes:
            failures.append(f"nodes={nodes} expected={list(case.expected_nodes)}")
        if case.expected_relations and list(case.expected_relations) != relations:
            failures.append(f"relations={relations} expected={list(case.expected_relations)}")
        if case.max_steps is not None and len(steps) > case.max_steps:
            failures.append(f"steps={len(steps)} exceeds max_steps={case.max_steps}")

        return ReasoningCaseResult(
            case=case,
            success=not failures,
            latency_ms=latency_ms,
            proof_steps=len(steps),
            trace=trace,
            failures=failures,
        )

    def evaluate_cases(self, cases: Iterable[ReasoningBenchmarkCase]) -> ReasoningBenchmarkReport:
        return ReasoningBenchmarkReport(
            case_results=[self.evaluate_case(case) for case in cases]
        )
