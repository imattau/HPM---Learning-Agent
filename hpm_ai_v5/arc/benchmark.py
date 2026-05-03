"""Dataset-backed ARC benchmarking helpers."""

from __future__ import annotations

import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .solver import ArcSolver


@dataclass(frozen=True, slots=True)
class ArcTaskResult:
    task_id: str
    route: str
    accepted: bool
    correct: bool
    test_count: int
    final_output_shape: tuple[int, int] | None
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ArcFullBenchmarkResult:
    result: str
    reason: str
    task_count: int
    solved_count: int
    accepted_count: int
    accuracy: float
    acceptance_rate: float
    failure_count: int
    task_results: tuple[ArcTaskResult, ...]


@dataclass(frozen=True, slots=True)
class ArcFullBenchmark:
    data_dir: Path = Path(__file__).resolve().parents[2] / "data" / "ARC-AGI-2" / "data" / "training"

    def run(self, *, limit: int | None = None, timeout_seconds: float = 2.0) -> ArcFullBenchmarkResult:
        files = sorted(self.data_dir.glob("*.json"))
        if limit is not None:
            files = files[:limit]

        solver = ArcSolver()
        task_results: list[ArcTaskResult] = []
        solved_count = 0
        accepted_count = 0
        failure_count = 0

        for path in files:
            raw_task = json.loads(path.read_text())
            try:
                if timeout_seconds > 0:
                    def _timeout_handler(_signum, _frame) -> None:
                        raise TimeoutError(f"ARC solve exceeded {timeout_seconds}s for {path.name}")

                    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                else:
                    previous_handler = None

                try:
                    packet = solver.solve(raw_task)
                finally:
                    if timeout_seconds > 0:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        if previous_handler is not None:
                            signal.signal(signal.SIGALRM, previous_handler)

                accepted = bool(packet.context.get("arc", {}).get("accepted"))
                expected = None
                if raw_task.get("test"):
                    expected = raw_task["test"][0].get("output")
                correct = packet.final_output == expected
                solved_count += 1 if correct else 0
                accepted_count += 1 if accepted else 0
                task_results.append(
                    ArcTaskResult(
                        task_id=str(raw_task.get("task_id") or path.stem),
                        route=str(packet.context.get("arc", {}).get("route", "unknown")),
                        accepted=accepted,
                        correct=correct,
                        test_count=len(raw_task.get("test", [])),
                        final_output_shape=(
                            (len(packet.final_output), len(packet.final_output[0]))
                            if packet.final_output and isinstance(packet.final_output, list) and packet.final_output and isinstance(packet.final_output[0], list)
                            else None
                        ),
                        timed_out=False,
                    )
                )
            except TimeoutError as exc:
                failure_count += 1
                task_results.append(
                    ArcTaskResult(
                        task_id=str(raw_task.get("task_id") or path.stem),
                        route=f"timeout:{type(exc).__name__}",
                        accepted=False,
                        correct=False,
                        test_count=len(raw_task.get("test", [])),
                        final_output_shape=None,
                        timed_out=True,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive benchmark guard
                failure_count += 1
                task_results.append(
                    ArcTaskResult(
                        task_id=str(raw_task.get("task_id") or path.stem),
                        route=f"error:{type(exc).__name__}",
                        accepted=False,
                        correct=False,
                        test_count=len(raw_task.get("test", [])),
                        final_output_shape=None,
                        timed_out=False,
                    )
                )

        task_count = len(files)
        accuracy = solved_count / max(1, task_count)
        acceptance_rate = accepted_count / max(1, task_count)
        if failure_count:
            result = "partial"
            reason = f"{failure_count} task(s) raised exceptions"
        else:
            result = "success"
            reason = "processed full ARC training split"
        return ArcFullBenchmarkResult(
            result=result,
            reason=reason,
            task_count=task_count,
            solved_count=solved_count,
            accepted_count=accepted_count,
            accuracy=accuracy,
            acceptance_rate=acceptance_rate,
            failure_count=failure_count,
            task_results=tuple(task_results),
        )
