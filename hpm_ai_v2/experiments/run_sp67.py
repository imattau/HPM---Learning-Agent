"""
run_sp67.py — SP67 experiment ported to hpm_ai_v2 agent layer.

Demonstrates that SocialAnalogicalAgent can solve the same curriculum as the
original SP67 experiment (mastery-driven study phase) using the new
domain-agnostic mixin architecture.

Curriculum:
  Phase 1: Scalar tasks (identity, const, double)
  Phase 2: List tasks (map-double, filter-positive)
  Phase 3: Cross-domain transfer test (novel scalar + novel list)
  Phase 4: Social sharing between two agents + recombination test

Run:
    PYTHONPATH=. python3 hpm_ai_v2/experiments/run_sp67.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hpm_ai_v2.agents.agents import SocialAnalogicalAgent, InducedSchemaAgent
from hpm_ai_v2.agents.mixins.social import SocialForest
from hpm_ai_v2.domains.list_domain import ListDomainConfig

# ---------------------------------------------------------------------------
# Curriculum tasks
# ---------------------------------------------------------------------------

SCALAR_TASKS = [
    # (task_id, goal_type, inputs, outputs)
    ("identity",  "scalar", [1, 2, 3, 5],    [1, 2, 3, 5]),
    ("const_one", "scalar", [1, 2, 3, 5],    [1, 1, 1, 1]),
    ("double",    "scalar", [1, 2, 3, 5],    [2, 4, 6, 10]),
]

LIST_TASKS = [
    ("map_double",      "map",    [[1,2,3], [4,5,6]], [[2,4,6], [8,10,12]]),
    ("filter_positive", "filter", [[1,-2,3,0,-5], [2,-1,0]], [[1,3], [2]]),
]

TRANSFER_TASKS = [
    ("triple",          "scalar", [1, 2, 3],          [3, 6, 9]),
    ("map_identity",    "map",    [[1,2], [3,4]],      [[1,2], [3,4]]),
]


def run_phase(label: str, agent: SocialAnalogicalAgent, tasks: list) -> int:
    """Run a list of tasks through the agent. Returns number solved."""
    solved = 0
    for task_id, goal_type, inputs, outputs in tasks:
        success, code, strategy = agent.solve(
            inputs=inputs,
            outputs=outputs,
            goal_type=goal_type,
            task_id=task_id,
        )
        status = "[OK]" if success else "[--]"
        print(f"  {status} {task_id:20s}  strategy={strategy}")
        if success:
            solved += 1
    return solved


def main() -> None:
    print("=" * 60)
    print("SP67 — hpm_ai_v2 SocialAnalogicalAgent")
    print("=" * 60)

    config = ListDomainConfig()
    shared = SocialForest(D=config.m_dim, cold_dir=Path("data/knowledge_base/sp67_social"))

    agent_a = SocialAnalogicalAgent(
        config=config,
        cold_dir="data/knowledge_base/sp67_a",
        social_forest=shared,
    )
    agent_b = SocialAnalogicalAgent(
        config=config,
        cold_dir="data/knowledge_base/sp67_b",
        social_forest=shared,
    )

    # Phase 1: scalar tasks (agent A)
    print("\nPhase 1 — Scalar tasks (Agent A)")
    p1 = run_phase("Phase 1", agent_a, SCALAR_TASKS)
    print(f"  -> {p1}/{len(SCALAR_TASKS)} solved")

    # Phase 2: list tasks (agent A)
    print("\nPhase 2 — List tasks (Agent A)")
    p2 = run_phase("Phase 2", agent_a, LIST_TASKS)
    print(f"  -> {p2}/{len(LIST_TASKS)} solved")

    # Phase 3: cross-domain transfer (agent A)
    print("\nPhase 3 — Transfer tasks (Agent A)")
    p3 = run_phase("Phase 3", agent_a, TRANSFER_TASKS)
    print(f"  -> {p3}/{len(TRANSFER_TASKS)} solved")

    # Phase 4: agent B gets social patterns from A, then tries transfer
    print("\nPhase 4 — Social sharing: Agent A -> Agent B")
    for name in list(agent_a.patterns.keys())[:3]:
        agent_a.share_pattern(name)
        print(f"  Shared pattern: {name}")

    print("\nPhase 4 — Transfer tasks (Agent B, after social sharing)")
    p4 = run_phase("Phase 4", agent_b, TRANSFER_TASKS)
    print(f"  -> {p4}/{len(TRANSFER_TASKS)} solved")

    # Summary
    print("\n" + "=" * 60)
    print("Meta-patterns learned by Agent A:")
    for pattern in agent_a.meta.meta_patterns():
        print(f"  {pattern}")

    total = p1 + p2 + p3 + p4
    possible = len(SCALAR_TASKS) + len(LIST_TASKS) + len(TRANSFER_TASKS) * 2
    print(f"\nTotal solved: {total}/{possible}")
    print("=" * 60)


if __name__ == "__main__":
    main()
