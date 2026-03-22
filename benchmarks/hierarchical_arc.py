"""
Hierarchical ARC Benchmark
===========================
Tests whether a 3-level StackedOrchestrator shapes L1 agent representations
better than a flat single-agent baseline on ARC discrimination tasks.

Stack config:
  L1: 2 agents, GaussianPattern, feature_dim=64, K=1 (always steps)
  L2: 2 agents, GaussianPattern, feature_dim=66, K=3 (fires every 3 L1 steps)
  L3: 1 agent,  GaussianPattern, feature_dim=68, K=3 (fires every 9 L1 steps)

Scoring: L1 agents (dim=64) score candidates via ensemble_score, same as the
flat baseline. L3 agents operate in 68-dim bundle space and cannot score 64-dim
raw ARC vectors directly. This benchmark tests whether hierarchical pressure
from L2/L3 improves L1 representation quality.

Both hierarchical and flat orchestrators are reset per task (same as multi_agent_arc.py).

Run:
    python benchmarks/hierarchical_arc.py

Note: Downloads ARC dataset from HuggingFace on first run (~5MB, cached).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.multi_agent_arc import (
    encode_pair, ensemble_score, load_tasks, task_fits, TRAIN_REPS, N_DISTRACTORS,
)
from benchmarks.multi_agent_common import make_orchestrator, print_results_table
from hpm.agents.stacked import LevelConfig, make_stacked_orchestrator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
L1_FEATURE_DIM = 64
STACK_CONFIGS = [
    LevelConfig(n_agents=2, K=1),   # L1: 64-dim
    LevelConfig(n_agents=2, K=3),   # L2: 66-dim, fires every 3 L1 steps
    LevelConfig(n_agents=1, K=3),   # L3: 68-dim, fires every 9 L1 steps
]
# With TRAIN_REPS=10 per pair and K_L2=3: L2 fires ~3 times per pair.
# With K_L3=3: L3 fires once per 9 L1 steps → fires at step 9 of training.
# For tasks with ≥1 training pair (all ARC tasks), L3 fires at least once.


def _make_flat_orchestrator():
    """Single flat agent baseline."""
    return make_orchestrator(
        n_agents=1,
        feature_dim=L1_FEATURE_DIM,
        agent_ids=["arc_flat"],
        pattern_types=["gaussian"],
    )


def run() -> dict:
    tasks = load_tasks()
    tasks = [t for t in tasks if task_fits(t)][:400]

    rng = np.random.default_rng(42)

    hierarchical_correct = 0
    flat_correct = 0
    n_evaluated = 0

    for i, task in enumerate(tasks):
        # Encode correct test output
        test_pair = task["test"][0]
        correct_vec = encode_pair(test_pair["input"], test_pair["output"])

        # Pick N_DISTRACTORS from other tasks
        distractor_idxs = [j for j in range(len(tasks)) if j != i]
        chosen = rng.choice(distractor_idxs, size=N_DISTRACTORS, replace=False)
        distractor_vecs = [
            encode_pair(tasks[di]["test"][0]["input"], tasks[di]["test"][0]["output"])
            for di in chosen
        ]
        candidates = [correct_vec] + distractor_vecs

        # --- Hierarchical ---
        stacked_orch, all_agents = make_stacked_orchestrator(
            l1_feature_dim=L1_FEATURE_DIM,
            level_configs=STACK_CONFIGS,
        )
        l1_agents = all_agents[0]

        for pair in task["train"]:
            vec = encode_pair(pair["input"], pair["output"])
            for _ in range(TRAIN_REPS):
                stacked_orch.step(vec)

        # Guard: L3 must have fired at least once for meaningful training
        if stacked_orch._level_ticks[-1] == 0:
            raise RuntimeError(
                f"L3 never fired on task {i} ({len(task['train'])} train pairs, "
                f"TRAIN_REPS={TRAIN_REPS}). Reduce K values or increase TRAIN_REPS."
            )

        # Score with L1 agents (dim=64 matches candidate vecs)
        scores_h = [ensemble_score(l1_agents, c) for c in candidates]
        if scores_h[0] == min(scores_h):  # lower NLL = more probable = correct
            hierarchical_correct += 1

        # --- Flat baseline ---
        flat_orch, flat_agents, _ = _make_flat_orchestrator()
        for pair in task["train"]:
            vec = encode_pair(pair["input"], pair["output"])
            for _ in range(TRAIN_REPS):
                flat_orch.step({flat_agents[0].agent_id: vec})

        scores_f = [ensemble_score(flat_agents, c) for c in candidates]
        if scores_f[0] == min(scores_f):
            flat_correct += 1

        n_evaluated += 1

    return {
        "n_tasks": n_evaluated,
        "hierarchical_correct": hierarchical_correct,
        "flat_correct": flat_correct,
        "hierarchical_acc": hierarchical_correct / n_evaluated if n_evaluated else 0.0,
        "flat_acc": flat_correct / n_evaluated if n_evaluated else 0.0,
        "chance": 1.0 / (N_DISTRACTORS + 1),
    }


def main():
    cfg_str = (
        f"3-level stack "
        f"(L1x2=64, L2x2=66 K={STACK_CONFIGS[1].K}, L3x1=68 K={STACK_CONFIGS[2].K})"
    )
    print(f"Running Hierarchical ARC Benchmark ({cfg_str})...")
    m = run()

    h_vs_chance = m["hierarchical_acc"] - m["chance"]
    f_vs_chance = m["flat_acc"] - m["chance"]
    h_vs_flat = m["hierarchical_acc"] - m["flat_acc"]

    print_results_table(
        title=f"Hierarchical ARC Benchmark ({m['n_tasks']} tasks, scored via L1 agents)",
        cols=["Setup", "Accuracy", "vs Chance", "vs Flat"],
        rows=[
            {
                "Setup": "Flat single-agent",
                "Accuracy": f"{m['flat_acc']:.1%}",
                "vs Chance": f"{f_vs_chance:+.1%}",
                "vs Flat": "—",
            },
            {
                "Setup": cfg_str,
                "Accuracy": f"{m['hierarchical_acc']:.1%}",
                "vs Chance": f"{h_vs_chance:+.1%}",
                "vs Flat": f"{h_vs_flat:+.1%}",
            },
        ],
    )


if __name__ == "__main__":
    main()
