"""Open adapter discovery benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from ..agents.open_adapter_discovery import (
    OpenAdapterDiscoveryAgent,
    OpenAdapterDiscoveryResult as AgentOpenAdapterDiscoveryResult,
    OpenAdapterTask,
    OpenAdapterTaskResult,
)


@dataclass(frozen=True, slots=True)
class OpenAdapterDiscoveryBenchmarkResult:
    result: str
    reason: str
    learned_profiles: int
    support_mean_accuracy: float
    query_mean_accuracy: float
    task_results: tuple[OpenAdapterTaskResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class OpenAdapterDiscoveryBenchmark:
    """Test whether the system can discover the right adapter path or defer."""

    agent: OpenAdapterDiscoveryAgent = field(default_factory=OpenAdapterDiscoveryAgent)

    def run(self) -> OpenAdapterDiscoveryBenchmarkResult:
        result: AgentOpenAdapterDiscoveryResult = self.agent.run()
        return OpenAdapterDiscoveryBenchmarkResult(
            result=result.result,
            reason=result.reason,
            learned_profiles=result.learned_profiles,
            support_mean_accuracy=result.support_mean_accuracy,
            query_mean_accuracy=result.query_mean_accuracy,
            task_results=result.task_results,
            trace=result.trace,
        )
