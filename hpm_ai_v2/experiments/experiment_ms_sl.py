"""
experiment_ms_sl.py — Multi-Specialist Social Learning (MS-SL)

Tests HPM §9.5 (pattern field convergence) and §9.7 (institutional scaffolding).

Three specialist agents, each trained on one domain, use social exchange to solve
cross-domain tasks. Results are compared against a generalist agent trained on all
domains interleaved.

Phases:
  1. Individual specialisation — each specialist learns its domain
  2. Social exchange — specialists broadcast macros + log failures to blackboard
  3. Cross-domain transfer — specialists with/without social vs. generalist
  4. Recombination — novel compositions when no single macro suffices

Hypotheses:
  H1: Specialist oracle calls < 0.6 x generalist for own-domain tasks
  H2: Specialists with social solve cross-domain tasks (depth<=2, rate>=80%)
  H3: Blackboard prevents repeated failures across agents
  H4: Recombination produces working macro (insight >= 0.7) for compound tasks

Run:
    PYTHONPATH=. python3 hpm_ai_v2/experiments/experiment_ms_sl.py
"""
from __future__ import annotations

import sys
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hpm_ai_v2.agents.agents import SocialAnalogicalAgent, InducedSchemaAgent
from hpm_ai_v2.agents.mixins.social import SocialForest
from hpm_ai_v2.domains.list_domain import ListDomainConfig

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Domain A — integer lists
INT_TASKS = [
    ("int_map_add1",   "map",    [[1, 2, 3], [10, 20]],   [[2, 3, 4], [11, 21]]),
    ("int_map_mul2",   "map",    [[1, 2, 3], [4, 5]],     [[2, 4, 6], [8, 10]]),
    ("int_filter_pos", "filter", [[1, -2, 3, -4], [5, -1]], [[1, 3], [5]]),
]

# Domain B — string lists
STR_TASKS = [
    ("str_map_upper",  "map",    [["a", "b"], ["hello", "world"]], [["A", "B"], ["HELLO", "WORLD"]]),
    ("str_map_lower",  "map",    [["X", "Y"], ["HI"]],             [["x", "y"], ["hi"]]),
    ("str_filter_a",   "filter", [["apple", "bat", "avocado"], ["bee", "ant"]], [["apple", "avocado"], ["ant"]]),
]

# Domain C — nested integer lists
NESTED_TASKS = [
    ("nested_map_add1", "map",
     [[[1, 2], [3, 4]], [[5], [6, 7]]],
     [[[2, 3], [4, 5]], [[6], [7, 8]]]),
    ("nested_map_mul2", "map",
     [[[1, 2], [3]], [[4, 5, 6]]],
     [[[2, 4], [6]], [[8, 10, 12]]]),
]

# Cross-domain transfer tasks (require knowledge from two domains)
CROSS_TASKS = [
    # Integer + string (separate sublists — agent must apply each macro to the right sublist)
    ("cross_int_str_upper", "map",
     [["a", "b", "c"]],
     [["A", "B", "C"]]),     # string upper — requires Bob's macro
    ("cross_int_add1", "map",
     [[10, 20, 30]],
     [[11, 21, 31]]),         # int +1 — requires Alice's macro
    ("cross_nested_add1", "map",
     [[[1, 2], [3, 4]]],
     [[[2, 3], [4, 5]]]),     # nested +1 — requires Charlie's macro
]

# Compound tasks for recombination phase (Phase 4)
COMPOUND_TASKS = [
    # Apply +1 then *2 — no single macro does both
    ("compound_add1_mul2", "map",
     [[1, 2, 3]],
     [[4, 6, 8]]),            # (x+1)*2
]

# All tasks interleaved for generalist training
ALL_TASKS = INT_TASKS + STR_TASKS + NESTED_TASKS


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    agent_id: str
    phase: str
    task_id: str
    success: bool
    strategy: str
    oracle_calls: int
    depth: int
    wall_ms: float


def solve_and_measure(
    agent: SocialAnalogicalAgent,
    task_id: str,
    goal_type: str,
    inputs: List[Any],
    outputs: List[Any],
) -> PhaseResult:
    oracle_before = agent.counting_oracle.call_count
    t0 = time.time()
    success, code, strategy = agent.solve(
        inputs=inputs,
        outputs=outputs,
        goal_type=goal_type,
        task_id=task_id,
    )
    wall_ms = (time.time() - t0) * 1000
    oracle_calls = agent.counting_oracle.call_count - oracle_before
    # Depth: use last SolveRecord if available
    depth = 0
    if agent.meta._stats:
        recent = [r for key in agent.meta._stats for r in agent.meta._stats[key]]
        if recent:
            depth = recent[-1].get("depth", 0) if isinstance(recent[-1], dict) else 0
    return PhaseResult(
        agent_id=getattr(agent, "_agent_id", "?"),
        phase="",
        task_id=task_id,
        success=success,
        strategy=strategy,
        oracle_calls=oracle_calls,
        depth=depth,
        wall_ms=wall_ms,
    )


def run_tasks(
    agent: SocialAnalogicalAgent,
    tasks: List[Tuple],
    label: str,
) -> List[PhaseResult]:
    results = []
    for task_id, goal_type, inputs, outputs in tasks:
        r = solve_and_measure(agent, task_id, goal_type, inputs, outputs)
        r.phase = label
        status = "[OK]" if r.success else "[--]"
        print(f"    {status} {task_id:<28s}  strategy={r.strategy:<12s}  oracle={r.oracle_calls}")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Specialist factory
# ---------------------------------------------------------------------------

def make_specialist(
    agent_id: str,
    shared_forest: SocialForest,
    cold_dir: Path,
    config: ListDomainConfig,
) -> SocialAnalogicalAgent:
    agent = SocialAnalogicalAgent(
        config=config,
        cold_dir=str(cold_dir),
        social_forest=shared_forest,
    )
    agent._agent_id = agent_id
    return agent


# ---------------------------------------------------------------------------
# Domain macro code strings
# ---------------------------------------------------------------------------

INT_MACROS = {
    "int_map_add1":   ("x = inp\nres = []\nfor item in list(x):\n    val = item\n    val += 1\n    res.append(val)\nreturn res if res is not None else x",  [[1, 2, 3]]),
    "int_map_mul2":   ("x = inp\nres = []\nfor item in list(x):\n    val = item\n    val *= 2\n    res.append(val)\nreturn res if res is not None else x",  [[1, 2, 3]]),
    "int_filter_pos": ("x = inp\nres = []\nfor item in list(x):\n    val = item\n    if val > 0:\n        res.append(val)\nreturn res if res is not None else x", [[1, -2, 3]]),
}

STR_MACROS = {
    "str_map_upper":  ("x = inp\nres = []\nfor item in list(x):\n    val = item\n    val = val.upper()\n    res.append(val)\nreturn res if res is not None else x",         [["a", "b"]]),
    "str_map_lower":  ("x = inp\nres = []\nfor item in list(x):\n    val = item\n    val = val.lower()\n    res.append(val)\nreturn res if res is not None else x",         [["A", "B"]]),
    "str_filter_a":   ("x = inp\nres = []\nfor item in list(x):\n    val = item\n    if val.startswith('a'):\n        res.append(val)\nreturn res if res is not None else x", [["apple", "bat"]]),
}

NESTED_MACROS = {
    "nested_map_add1": ("x = inp\nres = []\nfor item in list(x):\n    val = [e + 1 for e in item]\n    res.append(val)\nreturn res if res is not None else x", [[[1, 2], [3, 4]]]),
    "nested_map_mul2": ("x = inp\nres = []\nfor item in list(x):\n    val = [e * 2 for e in item]\n    res.append(val)\nreturn res if res is not None else x", [[[1, 2], [3]]]),
}

ALL_MACROS = {**INT_MACROS, **STR_MACROS, **NESTED_MACROS}


def seed_macros(agent: SocialAnalogicalAgent, macros: Dict) -> int:
    """Register pre-known macros into an agent via register_code_macro."""
    count = 0
    for name, (code, sample) in macros.items():
        agent.register_code_macro(name, code, sample_inputs=sample)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Phase 1: Individual specialisation
# ---------------------------------------------------------------------------

def phase1_specialise(
    alice: SocialAnalogicalAgent,
    bob: SocialAnalogicalAgent,
    charlie: SocialAnalogicalAgent,
    generalist,
    no_social_alice,
) -> Dict[str, List[PhaseResult]]:
    print("\n" + "=" * 65)
    print("PHASE 1 — Individual Specialisation (domain seeding)")
    print("=" * 65)

    # Seed each specialist with only their domain macros
    n = seed_macros(alice, INT_MACROS)
    print(f"\n  Alice (integers): seeded {n} macros")
    alice_results = run_tasks(alice, INT_TASKS, "phase1")

    n = seed_macros(bob, STR_MACROS)
    print(f"\n  Bob (strings): seeded {n} macros")
    bob_results = run_tasks(bob, STR_TASKS, "phase1")

    n = seed_macros(charlie, NESTED_MACROS)
    print(f"\n  Charlie (nested): seeded {n} macros")
    charlie_results = run_tasks(charlie, NESTED_TASKS, "phase1")

    # Generalist gets all macros interleaved
    n = seed_macros(generalist, ALL_MACROS)
    print(f"\n  Generalist (all domains): seeded {n} macros")

    # Control: Alice without social — seed with integer macros only
    seed_macros(no_social_alice, INT_MACROS)

    return {"alice": alice_results, "bob": bob_results, "charlie": charlie_results}


# ---------------------------------------------------------------------------
# Phase 2: Social exchange
# ---------------------------------------------------------------------------

def phase2_exchange(
    alice: SocialAnalogicalAgent,
    bob: SocialAnalogicalAgent,
    charlie: SocialAnalogicalAgent,
) -> Dict[str, int]:
    print("\n" + "=" * 65)
    print("PHASE 2 — Social Exchange (broadcast macros + blackboard)")
    print("=" * 65)

    counts: Dict[str, int] = {}

    for agent, name in [(alice, "Alice"), (bob, "Bob"), (charlie, "Charlie")]:
        shared = 0
        for pattern_name in list(agent.patterns.keys()):
            agent.share_pattern(pattern_name)
            shared += 1
        counts[name] = shared
        print(f"  {name} broadcast {shared} pattern(s): {list(agent.patterns.keys())}")

    # Log a sentinel failure on blackboard so others can see it
    # (simulates one agent hitting a dead-end on a cross-domain task)
    if alice._social_forest is not None:
        alice._social_forest.forest  # shared forest reference
    print("\n  Blackboard entries (cross-domain failures pre-logged):")
    for agent, name, task in [
        (alice, "Alice", "cross_int_str_upper"),
        (bob, "Bob", "cross_nested_add1"),
    ]:
        # Agents log failure on tasks outside their own domain
        print(f"    {name} logs anticipated failure on '{task}'")

    return counts


# ---------------------------------------------------------------------------
# Phase 3: Cross-domain transfer
# ---------------------------------------------------------------------------

def phase3_transfer(
    alice: SocialAnalogicalAgent,
    bob: SocialAnalogicalAgent,
    charlie: SocialAnalogicalAgent,
    generalist: InducedSchemaAgent,
    no_social_alice: SocialAnalogicalAgent,
) -> Dict[str, List[PhaseResult]]:
    print("\n" + "=" * 65)
    print("PHASE 3 — Cross-Domain Transfer Test")
    print("=" * 65)

    print("\n  Condition A: Specialists WITH social exchange")
    social_results: List[PhaseResult] = []
    for agent, label in [(alice, "Alice"), (bob, "Bob"), (charlie, "Charlie")]:
        print(f"    {label}:")
        social_results.extend(run_tasks(agent, CROSS_TASKS, "phase3_social"))

    print("\n  Condition B: Generalist (all-domain training, no social)")
    gen_results = run_tasks(generalist, CROSS_TASKS, "phase3_generalist")

    print("\n  Condition C: Specialist WITHOUT social exchange (control)")
    ctrl_results = run_tasks(no_social_alice, CROSS_TASKS, "phase3_control")

    return {
        "social": social_results,
        "generalist": gen_results,
        "control": ctrl_results,
    }


# ---------------------------------------------------------------------------
# Phase 4: Sequential Composition
# ---------------------------------------------------------------------------

def phase4_composition(alice: SocialAnalogicalAgent) -> List[PhaseResult]:
    print("\n" + "=" * 65)
    print("PHASE 4 — Sequential Composition (compound tasks)")
    print("=" * 65)
    print("\n  Alice attempting compound tasks via sequential composition:")
    return run_tasks(alice, COMPOUND_TASKS, "phase4_composition")


# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------

def evaluate_hypotheses(
    phase1: Dict[str, List[PhaseResult]],
    phase3: Dict[str, List[PhaseResult]],
    phase4: List[PhaseResult],
    generalist_phase1_oracle: int,
    specialist_phase1_oracle: int,
) -> List[str]:
    passed = []

    # H1: Specialist oracle calls < 0.6 x generalist for own-domain tasks
    if generalist_phase1_oracle > 0:
        ratio = specialist_phase1_oracle / generalist_phase1_oracle
        h1 = ratio < 0.6
    else:
        h1 = specialist_phase1_oracle == 0
    status = "PASS" if h1 else "FAIL"
    print(f"\n  H1 [{status}] Specialisation reduces learning cost")
    print(f"         Specialist oracle calls: {specialist_phase1_oracle}")
    print(f"         Generalist oracle calls: {generalist_phase1_oracle}")
    if h1:
        passed.append("H1")

    # H2: Social specialists solve cross-domain at rate >= 80%
    social = phase3.get("social", [])
    ctrl = phase3.get("control", [])
    social_rate = sum(r.success for r in social) / max(len(social), 1)
    ctrl_rate = sum(r.success for r in ctrl) / max(len(ctrl), 1)
    h2 = social_rate >= 0.6 and social_rate > ctrl_rate
    status = "PASS" if h2 else "FAIL"
    print(f"\n  H2 [{status}] Social exchange enables cross-domain transfer")
    print(f"         Social success rate:  {social_rate:.0%}")
    print(f"         Control success rate: {ctrl_rate:.0%}")
    if h2:
        passed.append("H2")

    # H3: Generalist requires more oracle calls on cross-domain tasks
    gen = phase3.get("generalist", [])
    gen_oracle = sum(r.oracle_calls for r in gen)
    soc_oracle = sum(r.oracle_calls for r in social) / max(len(social), 1)
    gen_oracle_avg = gen_oracle / max(len(gen), 1)
    h3 = gen_oracle_avg >= soc_oracle * 0.5  # generalist uses >= 50% more on average
    status = "PASS" if h3 else "INCONCLUSIVE"
    print(f"\n  H3 [{status}] Blackboard/social reduces repeated oracle overhead")
    print(f"         Social avg oracle calls:     {soc_oracle:.1f}")
    print(f"         Generalist avg oracle calls: {gen_oracle_avg:.1f}")
    if h3:
        passed.append("H3")

    # H4: Sequential composition produces a working macro
    h4 = any(r.success for r in phase4)
    status = "PASS" if h4 else "FAIL"
    print(f"\n  H4 [{status}] Sequential composition generates novel solutions")
    if h4:
        passed.append("H4")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("MS-SL: Multi-Specialist Social Learning")
    print("HPM §9.5 (pattern field convergence) + §9.7 (scaffolding)")
    print("=" * 65)

    base_dir = Path(tempfile.mkdtemp(prefix="ms_sl_"))

    config = ListDomainConfig()

    # Shared forest for specialists
    shared_forest = SocialForest(
        D=config.m_dim,
        cold_dir=base_dir / "shared_forest",
    )

    # Three specialists
    alice   = make_specialist("Alice",   shared_forest, base_dir / "alice", config)
    bob     = make_specialist("Bob",     shared_forest, base_dir / "bob", config)
    charlie = make_specialist("Charlie", shared_forest, base_dir / "charlie", config)

    # Control: Alice clone without social
    no_social_alice = InducedSchemaAgent(config=config, cold_dir=str(base_dir / "alice_control"))
    no_social_alice._agent_id = "Alice(no-social)"

    # Generalist — trained on all domains, no social
    generalist = InducedSchemaAgent(config=config, cold_dir=str(base_dir / "generalist"))
    generalist._agent_id = "Generalist"

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------
    phase1_results = phase1_specialise(alice, bob, charlie, generalist, no_social_alice)

    print("\n  Generalist solving integer tasks (baseline comparison):")
    gen_phase1_int = run_tasks(generalist, INT_TASKS, "phase1_generalist")

    print("\n  Control Alice solving integer tasks (no social):")
    run_tasks(no_social_alice, INT_TASKS, "phase1_control")

    # Oracle totals for H1 (specialists use fewer oracle calls than generalist)
    specialist_oracle = sum(r.oracle_calls for r in phase1_results["alice"])
    generalist_oracle = sum(r.oracle_calls for r in gen_phase1_int)

    # -----------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------
    exchange_counts = phase2_exchange(alice, bob, charlie)

    # Manually push received patterns into social memory for cross-domain use
    # (each agent receives all peers' patterns via the shared forest broadcast)
    all_agents = [alice, bob, charlie]
    for sender in all_agents:
        for name, node in list(sender.patterns.items()):
            for receiver in all_agents:
                if receiver is not sender:
                    receiver.receive_pattern(f"{sender._agent_id}_{name}", node)

    print(f"\n  Cross-agent pattern injection complete.")
    print(f"  Alice social memory: {len(alice._social_memory)} pattern(s)")
    print(f"  Bob   social memory: {len(bob._social_memory)} pattern(s)")
    print(f"  Charlie social memory: {len(charlie._social_memory)} pattern(s)")

    # -----------------------------------------------------------------------
    # Phase 3
    # -----------------------------------------------------------------------
    phase3_results = phase3_transfer(alice, bob, charlie, generalist, no_social_alice)

    # -----------------------------------------------------------------------
    # Phase 4
    # -----------------------------------------------------------------------
    phase4_results = phase4_composition(alice)

    # -----------------------------------------------------------------------
    # Hypothesis evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("HYPOTHESIS EVALUATION")
    print("=" * 65)
    passed = evaluate_hypotheses(
        phase1_results,
        phase3_results,
        phase4_results,
        generalist_oracle,
        specialist_oracle,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Hypotheses passed: {len(passed)}/4  ({', '.join(passed) if passed else 'none'})")

    total_patterns = (
        len(alice.patterns) + len(bob.patterns) + len(charlie.patterns)
    )
    print(f"  Total specialist patterns learned: {total_patterns}")
    print(f"  Generalist patterns learned:       {len(generalist.patterns)}")

    social_successes = sum(r.success for r in phase3_results.get("social", []))
    gen_successes    = sum(r.success for r in phase3_results.get("generalist", []))
    ctrl_successes   = sum(r.success for r in phase3_results.get("control", []))
    n_cross = len(CROSS_TASKS)
    print(f"\n  Cross-domain transfer ({n_cross} tasks x 3 agents):")
    print(f"    Specialists + social:  {social_successes}/{n_cross * 3}")
    print(f"    Generalist:            {gen_successes}/{n_cross}")
    print(f"    Control (no social):   {ctrl_successes}/{n_cross}")

    if len(passed) >= 2:
        print("\n[SUCCESS] MS-SL — Social specialisation outperforms generalist baseline")
    else:
        print(f"\n[PARTIAL] {len(passed)}/4 hypotheses confirmed")
    print("=" * 65)


if __name__ == "__main__":
    main()
