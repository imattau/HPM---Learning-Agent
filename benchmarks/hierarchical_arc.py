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
    """Single flat agent baseline (per-task reset)."""
    return make_orchestrator(
        n_agents=1,
        feature_dim=L1_FEATURE_DIM,
        agent_ids=["arc_flat"],
        pattern_types=["gaussian"],
    )


def _make_persistent_l1_orchestrator(store=None):
    """HPM config for persistent cross-task L1 learning.

    Key additions vs per-task reset:
    - kappa_d_levels: density bias stabilises recurring cross-task patterns
      (the critical mechanism — patterns that recur across tasks gain weight)
    - beta_comp=0.1: compression reward encourages abstraction over repetition
    - init_sigma=2.0: broader initial patterns for better cross-task coverage
    - gamma_soc=0.5: social learning between L1 agents via shared field

    Monitor/strategist omitted — they add large per-step overhead across 342 tasks.
    """
    return make_orchestrator(
        n_agents=2,
        feature_dim=L1_FEATURE_DIM,
        agent_ids=["l1_0", "l1_1"],
        pattern_types=["gaussian", "gaussian"],
        with_monitor=False,
        beta_comp=0.1,
        gamma_soc=0.5,
        init_sigma=2.0,
        kappa_d_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
        kappa_D=0.5,
        store=store,
    )


def _make_persistent_flat_orchestrator(store=None):
    """Persistent flat baseline — same config as L1 persistent for fair comparison."""
    return make_orchestrator(
        n_agents=1,
        feature_dim=L1_FEATURE_DIM,
        agent_ids=["arc_flat"],
        pattern_types=["gaussian"],
        with_monitor=False,
        beta_comp=0.1,
        gamma_soc=0.5,
        init_sigma=2.0,
        kappa_d_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
        kappa_D=0.5,
        store=store,
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


def run_persistent() -> dict:
    """Persistent mode: single StackedOrchestrator across all tasks with ContextualPatternStore.

    L1 agents use a ContextualPatternStore (wraps TieredStore):
      - Tier 1 (ephemeral): task-specific patterns, cleared at end of each task
      - Tier 2 (persistent): meta-patterns promoted from successful tasks
      - Archive: Tier 2 snapshot saved per task, warm-started from structurally similar past tasks

    L2/L3 use plain InMemoryStore — they accumulate cross-task signal through
    L1 bundles and are never reset.

    Flat baseline uses an identical ContextualPatternStore for fair comparison.
    CrossTaskRecombinator runs on L1 every 10 tasks to build higher-level patterns.
    """
    import os
    from hpm.store.tiered_store import TieredStore
    from hpm.store.contextual_store import ContextualPatternStore, extract_signature
    from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator
    from hpm.agents.stacked import StackedOrchestrator

    tasks = load_tasks()
    tasks = [t for t in tasks if task_fits(t)][:400]

    rng = np.random.default_rng(42)

    os.makedirs("data", exist_ok=True)
    l1_tiered = TieredStore()
    flat_tiered = TieredStore()

    # Build L1 orchestrators using bare TieredStores.
    # ContextualPatternStore wraps l1_tiered for lifecycle management (begin/end_context).
    # Agents use l1_tiered directly — ContextualPatternStore delegates all save/query calls to it.
    l1_orch, l1_agents, _ = _make_persistent_l1_orchestrator(store=l1_tiered)
    flat_orch, flat_agents, _ = _make_persistent_flat_orchestrator(store=flat_tiered)

    # Build L2/L3 via make_stacked_orchestrator, then assemble manually with L1.
    _upper_configs = STACK_CONFIGS[1:]  # L2, L3 configs
    _upper_orch, _upper_agents = make_stacked_orchestrator(
        l1_feature_dim=L1_FEATURE_DIM + 2,  # L2 input dim
        level_configs=_upper_configs,
    )

    level_orches = [l1_orch] + _upper_orch.level_orches
    level_agents = [l1_agents] + _upper_orch.level_agents
    level_Ks = [1] + [cfg.K for cfg in STACK_CONFIGS[1:]]
    stacked_orch = StackedOrchestrator(level_orches, level_agents, level_Ks)

    # Build ContextualPatternStores AFTER stacked_orch so we can pass l3_agents.
    # stacked_orch.level_agents[-1] = the L3 agent list.
    l1_contextual = ContextualPatternStore(
        tiered_store=l1_tiered,
        archive_dir="data/archives/hierarchical_arc_persistent",
        l3_agents=stacked_orch.level_agents[-1],
    )
    flat_contextual = ContextualPatternStore(
        tiered_store=flat_tiered,
        archive_dir="data/archives/hierarchical_arc_persistent_flat",
    )

    recombinator = CrossTaskRecombinator()

    hierarchical_correct = 0
    flat_correct = 0
    n_evaluated = 0

    for i, task in enumerate(tasks):
        train_pairs = task["train"]

        # Extract structural signature from first training pair's input grid.
        # Collect first 3 encoded training vecs for NLL fingerprinting (warm-start filter).
        first_grid = np.array(train_pairs[0]["input"], dtype=float)
        sig = extract_signature(first_grid)
        first_obs = [
            encode_pair(
                train_pairs[j % len(train_pairs)]["input"],
                train_pairs[j % len(train_pairs)]["output"],
            )
            for j in range(min(3, len(train_pairs)))
        ]

        l1_ctx_id = l1_contextual.begin_context(sig, first_obs)
        flat_ctx_id = flat_contextual.begin_context(sig, first_obs)

        # Train on this task's pairs
        for pair in train_pairs:
            vec = encode_pair(pair["input"], pair["output"])
            for _ in range(TRAIN_REPS):
                stacked_orch.step(vec)
                flat_orch.step({flat_agents[0].agent_id: vec})

        # Score
        test_pair = task["test"][0]
        correct_vec = encode_pair(test_pair["input"], test_pair["output"])
        distractor_idxs = [j for j in range(len(tasks)) if j != i]
        chosen = rng.choice(distractor_idxs, size=N_DISTRACTORS, replace=False)
        distractor_vecs = [
            encode_pair(tasks[di]["test"][0]["input"], tasks[di]["test"][0]["output"])
            for di in chosen
        ]
        candidates = [correct_vec] + distractor_vecs

        scores_h = [ensemble_score(l1_agents, c) for c in candidates]
        h_correct = scores_h[0] == min(scores_h)
        if h_correct:
            hierarchical_correct += 1

        scores_f = [ensemble_score(flat_agents, c) for c in candidates]
        f_correct = scores_f[0] == min(scores_f)
        if f_correct:
            flat_correct += 1

        # End context: archive Tier 2, promote patterns on success.
        # Lower similarity threshold (0.7 vs default 0.95) to build cross-task library
        # on ARC's deliberately diverse task distribution.
        l1_contextual.end_context(l1_ctx_id, success_metrics={"correct": h_correct},
                                  similarity_threshold=0.7)
        flat_contextual.end_context(flat_ctx_id, success_metrics={"correct": f_correct},
                                    similarity_threshold=0.7)

        n_evaluated += 1

        # Cross-task recombination every 10 tasks
        if (i + 1) % 10 == 0:
            for agent in l1_agents:
                recombinator.consolidate(l1_tiered, agent.agent_id)

    # Guard: L3 must have fired during the run
    if stacked_orch._level_ticks[-1] == 0:
        raise RuntimeError(
            "L3 never fired across the full task sequence. "
            "Reduce K values or increase TRAIN_REPS."
        )

    return {
        "n_tasks": n_evaluated,
        "hierarchical_correct": hierarchical_correct,
        "flat_correct": flat_correct,
        "hierarchical_acc": hierarchical_correct / n_evaluated if n_evaluated else 0.0,
        "flat_acc": flat_correct / n_evaluated if n_evaluated else 0.0,
        "chance": 1.0 / (N_DISTRACTORS + 1),
        "l3_ticks": stacked_orch._level_ticks[-1],
        "l1_tier2": len(l1_contextual.query_tier2("l1_0")),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent", action="store_true",
                        help="Single orchestrator across all tasks (cross-task learning)")
    args = parser.parse_args()

    cfg_str = (
        f"3-level stack "
        f"(L1x2=64, L2x2=66 K={STACK_CONFIGS[1].K}, L3x1=68 K={STACK_CONFIGS[2].K})"
    )

    if args.persistent:
        print(f"Running Hierarchical ARC Benchmark — persistent mode ({cfg_str})...")
        m = run_persistent()
        print(f"  L3 fired {m['l3_ticks']} times across {m['n_tasks']} tasks")
        print(f"  L1 Tier 2 patterns promoted: {m['l1_tier2']}")
    else:
        print(f"Running Hierarchical ARC Benchmark ({cfg_str})...")
        m = run()

    h_vs_chance = m["hierarchical_acc"] - m["chance"]
    f_vs_chance = m["flat_acc"] - m["chance"]
    h_vs_flat = m["hierarchical_acc"] - m["flat_acc"]

    mode = "persistent" if args.persistent else "per-task reset"
    print_results_table(
        title=f"Hierarchical ARC Benchmark ({m['n_tasks']} tasks, {mode}, scored via L1 agents)",
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
