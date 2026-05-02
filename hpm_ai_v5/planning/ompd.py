"""Online Meta-Pattern Discovery benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Mapping, Sequence

from ..agents import MetaPatternDiscoveryAgent


@dataclass(frozen=True, slots=True)
class OMPDTask:
    name: str
    items: tuple[str, ...]
    dependencies: dict[str, list[str]]
    rewards: dict[str, float]

    def optimal_plan(self) -> list[str]:
        return [f"collect_{item}" for item in self.items]

    def optimal_reward(self) -> float:
        return float(sum(self.rewards.get(item, 0.0) for item in self.items))

    def execute_plan(self, plan: Sequence[str]) -> float:
        expected = self.optimal_plan()
        if list(plan) == expected:
            return self.optimal_reward()
        matched = 0
        for left, right in zip(plan, expected):
            if left != right:
                break
            matched += 1
        if matched == 0:
            return 0.0
        return float(sum(self.rewards.get(item, 0.0) for item in self.items[:matched]))


@dataclass(frozen=True, slots=True)
class OMPDTaskResult:
    name: str
    plan: list[str]
    reward: float
    optimal_reward: float
    ratio: float
    confidence: float
    matched: bool
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OMPDResult:
    result: str
    reason: str
    meta_pattern_count: int
    training_mean_ratio: float
    test_ratio: float
    task_results: tuple[OMPDTaskResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class OnlineMetaPatternDiscoveryBenchmark:
    """Learn an abstract dependency chain from three tasks and transfer to a fourth."""

    agent: MetaPatternDiscoveryAgent = field(default_factory=MetaPatternDiscoveryAgent)
    training_tasks: tuple[OMPDTask, ...] = field(default_factory=lambda: (
        OMPDTask(
            name="Maze1",
            items=("key", "lever", "chest"),
            dependencies={"key": [], "lever": ["key"], "chest": ["lever"]},
            rewards={"key": 1.0, "lever": 2.0, "chest": 50.0},
        ),
        OMPDTask(
            name="Maze2",
            items=("token", "button", "vault"),
            dependencies={"token": [], "button": ["token"], "vault": ["button"]},
            rewards={"token": 5.0, "button": 1.0, "vault": 100.0},
        ),
        OMPDTask(
            name="Maze3",
            items=("gem", "pedestal", "portal"),
            dependencies={"gem": [], "pedestal": ["gem"], "portal": ["pedestal"]},
            rewards={"gem": 10.0, "pedestal": 1.0, "portal": 200.0},
        ),
    ))
    test_task: OMPDTask = field(default_factory=lambda: OMPDTask(
        name="Maze4",
        items=("rune", "altar", "gate"),
        dependencies={"rune": [], "altar": ["rune"], "gate": ["altar"]},
        rewards={"rune": 3.0, "altar": 4.0, "gate": 80.0},
    ))

    def _learn(self) -> list[OMPDTaskResult]:
        results: list[OMPDTaskResult] = []
        for task in self.training_tasks:
            decision = self.agent.observe(task.name, task.items, task.dependencies, task.rewards)
            reward = task.execute_plan(decision.plan)
            ratio = reward / task.optimal_reward() if task.optimal_reward() > 0.0 else 0.0
            results.append(
                OMPDTaskResult(
                    name=task.name,
                    plan=list(decision.plan),
                    reward=reward,
                    optimal_reward=task.optimal_reward(),
                    ratio=ratio,
                    confidence=decision.confidence,
                    matched=decision.matched,
                    trace={
                        "decision": decision.trace,
                        "signature": decision.signature,
                    },
                )
            )
        return results

    def _test(self) -> OMPDTaskResult:
        decision = self.agent.solve(self.test_task.name, self.test_task.items, self.test_task.dependencies, self.test_task.rewards)
        reward = self.test_task.execute_plan(decision.plan)
        optimal = self.test_task.optimal_reward()
        ratio = reward / optimal if optimal > 0.0 else 0.0
        return OMPDTaskResult(
            name=self.test_task.name,
            plan=list(decision.plan),
            reward=reward,
            optimal_reward=optimal,
            ratio=ratio,
            confidence=decision.confidence,
            matched=decision.matched,
            trace={
                "decision": decision.trace,
                "signature": decision.signature,
                "meta_pattern_count": len(self.agent.meta_patterns),
            },
        )

    def run(self) -> OMPDResult:
        training_results = self._learn()
        test_result = self._test()
        meta_pattern_count = len(self.agent.meta_patterns)
        training_mean_ratio = fmean(result.ratio for result in training_results) if training_results else 0.0
        passed = meta_pattern_count >= 1 and test_result.ratio >= 0.8 and test_result.confidence >= 0.8
        reason = "success" if passed else "meta-pattern discovery failed"
        return OMPDResult(
            result="success" if passed else "failure",
            reason=reason,
            meta_pattern_count=meta_pattern_count,
            training_mean_ratio=training_mean_ratio,
            test_ratio=test_result.ratio,
            task_results=tuple([*training_results, test_result]),
            trace={
                "training": [result.trace for result in training_results],
                "test": test_result.trace,
                "meta_patterns": list(self.agent.meta_patterns.keys()),
            },
        )
