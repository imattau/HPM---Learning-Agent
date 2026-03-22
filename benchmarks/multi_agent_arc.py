"""
Multi-Agent Benchmark 4: ARC (Abstraction and Reasoning Corpus)
================================================================
Same ARC discrimination task as arc_benchmark.py, but each task uses two
agents with partitioned training pairs:

  agent_a trains on even-indexed pairs (0, 2, 4, ...)
  agent_b trains on odd-indexed pairs  (1, 3, 5, ...)

Each agent sees fewer examples but learns from a different subset. Cross-agent
pattern sharing via the PatternField lets them benefit from each other's examples.
Scoring uses an ensemble: sum of per-agent weighted NLL scores.

For tasks with only 1 training pair, both agents train on it (no split possible).

Protocol:
  - Train 2 agents on partitioned (input, output) grid pairs (10 steps each)
  - Test: identify correct output from 4 distractors drawn from other tasks
  - Scoring: ensemble NLL = sum of per-agent weighted NLL scores
  - Metric: accuracy vs 20% chance baseline

Run:
    python benchmarks/multi_agent_arc.py

Note: Downloads ARC dataset from HuggingFace on first run (~5MB, cached).
Sign convention: GaussianPattern.log_prob returns NLL (lower = more probable).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.multi_agent_common import make_orchestrator, print_results_table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_GRID_DIM = 20
GRID_SIZE = MAX_GRID_DIM * MAX_GRID_DIM   # 400
RAW_DIM = GRID_SIZE * 2                   # 800 (full grid pair)
FEATURE_DIM = 64                          # projected dimension (JL projection)
TRAIN_REPS = 10
N_DISTRACTORS = 4

# Fixed random projection matrix: RAW_DIM → FEATURE_DIM
# Same matrix across all tasks so distances are consistent.
_PROJ = np.random.default_rng(0).standard_normal((RAW_DIM, FEATURE_DIM)) / np.sqrt(FEATURE_DIM)


def encode_grid(grid):
    flat = []
    for row in grid:
        for val in row:
            flat.append(float(val) / 9.0)
    flat.extend([0.0] * (GRID_SIZE - len(flat)))
    return np.array(flat[:GRID_SIZE], dtype=float)


def encode_pair(input_grid, output_grid):
    raw = np.concatenate([encode_grid(input_grid), encode_grid(output_grid)])
    return raw @ _PROJ  # project 800 → 64


def grid_fits(grid):
    if len(grid) > MAX_GRID_DIM:
        return False
    return all(len(row) <= MAX_GRID_DIM for row in grid)


def task_fits(task):
    for pair in task["train"]:
        if not grid_fits(pair["input"]) or not grid_fits(pair["output"]):
            return False
    for pair in task["test"]:
        if not grid_fits(pair["input"]) or not grid_fits(pair["output"]):
            return False
    return True


def _to_plain(obj):
    """Recursively convert HuggingFace dataset objects to plain Python."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def load_tasks():
    from datasets import load_dataset
    ds = load_dataset("lordspline/arc-agi", split="training")
    return [_to_plain({"train": item["train"], "test": item["test"]}) for item in ds]


def ensemble_score(agents, vec: np.ndarray) -> float:
    """
    Compute ensemble score for a candidate vector.

    Positive patterns contribute to the total (higher NLL = less probable = worse).
    Negative patterns are subtracted: a candidate close to a taboo pattern has low
    NLL under that pattern, so subtracting a small value leaves the score HIGH (worse).
    A candidate far from taboo patterns has high NLL, so subtracting a large value
    makes the score LOWER (better rank).

    Formula:
        total += pos_weight * pos_NLL   (existing behaviour)
        total -= neg_weight * neg_NLL   (new inhibitory term)

    Lower total = more preferred candidate (consistent with existing ranking).
    Returns 0.0 if all stores are empty (backward compatible).

    Sign convention: GaussianPattern.log_prob returns NLL (lower = more probable).
    """
    total = 0.0
    any_records = False
    for agent in agents:
        pos = agent.store.query(agent.agent_id)
        if pos:
            any_records = True
            total += sum(w * p.log_prob(vec) for p, w in pos)
        if hasattr(agent.store, 'query_negative'):
            neg = agent.store.query_negative(agent.agent_id)
            if neg:
                any_records = True
                total -= sum(w * p.log_prob(vec) for p, w in neg)
    return total if any_records else 0.0


def make_arc_orchestrator(pattern_types=None):
    """Fresh 2-agent HPM orchestrator configured for ARC (per-task reset)."""
    return make_orchestrator(
        n_agents=2,
        feature_dim=FEATURE_DIM,
        agent_ids=["arc_a", "arc_b"],
        with_monitor=False,   # skip monitor for speed (per-task reset)
        beta_comp=0.0,
        gamma_soc=0.5,        # enable social learning via shared field
        T_recomb=5,
        recomb_cooldown=3,
        init_sigma=2.0,
        pattern_types=pattern_types or ["gaussian", "gaussian"],
    )


def make_arc_persistent_orchestrator(store=None):
    """Single long-lived orchestrator for persistent cross-task learning.

    Full HPM stack enabled so patterns can level up across tasks:
    - StructuralLawMonitor tracks diversity/redundancy every 20 steps
    - RecombinationStrategist triggers bursts when diversity collapses
    - min_recomb_level=1 allows recombination before patterns reach L4
    - beta_comp rewards compression (encourages hierarchical abstraction)
    - kappa_d_levels active so density bias stabilises recurring patterns

    Args:
        store: Optional PatternStore to inject (e.g. TieredStore). If None,
               defaults to InMemoryStore().
    """
    return make_orchestrator(
        n_agents=2,
        feature_dim=FEATURE_DIM,
        agent_ids=["arc_a", "arc_b"],
        with_monitor=True,
        T_monitor=20,
        beta_comp=0.1,
        gamma_soc=0.5,
        T_recomb=5,
        recomb_cooldown=3,
        init_sigma=2.0,
        min_recomb_level=1,
        kappa_d_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
        kappa_D=0.5,
        store=store,
    )


def make_arc_laplace_orchestrator():
    """Fresh 2-agent HPM orchestrator using LaplacePattern for ARC."""
    return make_arc_orchestrator(pattern_types=["laplace", "laplace"])


def evaluate_task(task, all_tasks, task_idx, pattern_types=None):
    """
    Run one ARC task with 2-agent ensemble scoring.

    Training is interleaved through orch.step() so the PatternField
    broadcasts patterns between agents as they learn from their
    respective partitions. Each step both agents observe their current
    pair simultaneously, with the shorter list cycled to match.

    Returns:
        (correct, rank)
    """
    orch, agents, store = make_arc_orchestrator(pattern_types=pattern_types)

    # Partition training pairs between agents.
    # agent_a gets even-indexed pairs, agent_b gets odd-indexed pairs.
    # If only 1 pair exists, both agents train on it.
    train_pairs = task["train"]
    pairs_a = train_pairs[0::2] or train_pairs
    pairs_b = train_pairs[1::2] or train_pairs

    # Build interleaved schedule: cycle shorter list to match longer.
    # Each entry is (obs_a, obs_b) for one orch.step() call.
    n = max(len(pairs_a), len(pairs_b))
    schedule = [
        (
            encode_pair(pairs_a[i % len(pairs_a)]["input"], pairs_a[i % len(pairs_a)]["output"]),
            encode_pair(pairs_b[i % len(pairs_b)]["input"], pairs_b[i % len(pairs_b)]["output"]),
        )
        for i in range(n)
    ]

    # Train via orchestrator so PatternField broadcasts fire between agents.
    for _ in range(TRAIN_REPS):
        for obs_a, obs_b in schedule:
            orch.step({"arc_a": obs_a, "arc_b": obs_b})

    # Evaluation phase
    test_pair = task["test"][0]
    test_input = test_pair["input"]
    correct_output = test_pair["output"]

    correct_vec = encode_pair(test_input, correct_output)

    rng = np.random.default_rng(task_idx)
    other_indices = [j for j in range(len(all_tasks)) if j != task_idx and all_tasks[j]["train"]]
    distractor_indices = rng.choice(other_indices, size=N_DISTRACTORS, replace=False)
    distractor_vecs = [
        encode_pair(test_input, all_tasks[j]["train"][0]["output"])
        for j in distractor_indices
    ]

    correct_score = ensemble_score(agents, correct_vec)
    distractor_scores = [ensemble_score(agents, v) for v in distractor_vecs]

    rank = 1 + sum(1 for s in distractor_scores if s <= correct_score)
    correct = correct_score < min(distractor_scores)

    return correct, rank


_worker_all_tasks = None


def _init_worker(all_tasks):
    global _worker_all_tasks
    _worker_all_tasks = all_tasks


def _eval_worker(args):
    task, task_idx = args
    return evaluate_task(task, _worker_all_tasks, task_idx)


def run_persistent(eligible, all_tasks, use_contextual=True):
    """
    Persistent mode: single orchestrator across all tasks.
    Uses TieredStore so task-specific patterns are ephemeral (Tier 1) and
    successful cross-task meta-patterns are promoted to persistent Tier 2.
    CrossTaskRecombinator runs every 10 tasks to build higher-level meta-patterns.
    Sequential (no parallelism — shared mutable store).

    Args:
        use_contextual: If True (default), wraps TieredStore in ContextualPatternStore
            for structural warm-starting from similar past tasks. If False, uses bare
            TieredStore (original behaviour, useful for comparison).
    """
    from hpm.store.tiered_store import TieredStore
    from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator

    tiered = TieredStore()

    if use_contextual:
        from hpm.store.contextual_store import ContextualPatternStore, extract_signature
        archive_dir = "data/archives/arc_persistent"
        os.makedirs("data", exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)
        store_obj = ContextualPatternStore(
            tiered_store=tiered,
            archive_dir=archive_dir,
        )
    else:
        store_obj = tiered

    orch, agents, store = make_arc_persistent_orchestrator(store=store_obj)
    recombinator = CrossTaskRecombinator()

    correct_count = 0
    rank_sum = 0

    for run_idx, (task_idx, task) in enumerate(eligible):
        # Train on this task's pairs (adds to Tier 1 of TieredStore)
        train_pairs = task["train"]
        pairs_a = train_pairs[0::2] or train_pairs
        pairs_b = train_pairs[1::2] or train_pairs

        n = max(len(pairs_a), len(pairs_b))
        schedule = [
            (
                encode_pair(pairs_a[i % len(pairs_a)]["input"], pairs_a[i % len(pairs_a)]["output"]),
                encode_pair(pairs_b[i % len(pairs_b)]["input"], pairs_b[i % len(pairs_b)]["output"]),
            )
            for i in range(n)
        ]

        if use_contextual:
            # Extract structural signature from first training pair's input grid
            first_grid = np.array(train_pairs[0]["input"], dtype=float)
            sig = extract_signature(first_grid)
            # Collect first 3 encoded training pair vectors for NLL fingerprinting
            all_vecs = [
                encode_pair(pairs_a[i % len(pairs_a)]["input"], pairs_a[i % len(pairs_a)]["output"])
                for i in range(min(3, n))
            ]
            context_id = store_obj.begin_context(sig, all_vecs)
        else:
            context_id = str(task_idx)
            tiered.begin_context(context_id)

        for _ in range(TRAIN_REPS):
            for obs_a, obs_b in schedule:
                orch.step({"arc_a": obs_a, "arc_b": obs_b})

        # Evaluate against distractors
        test_pair = task["test"][0]
        correct_vec = encode_pair(test_pair["input"], test_pair["output"])
        rng = np.random.default_rng(task_idx)
        other_indices = [j for j in range(len(all_tasks)) if j != task_idx and all_tasks[j]["train"]]
        distractor_indices = rng.choice(other_indices, size=N_DISTRACTORS, replace=False)
        distractor_vecs = [
            encode_pair(test_pair["input"], all_tasks[j]["train"][0]["output"])
            for j in distractor_indices
        ]

        correct_score = ensemble_score(agents, correct_vec)
        distractor_scores = [ensemble_score(agents, v) for v in distractor_vecs]
        rank = 1 + sum(1 for s in distractor_scores if s <= correct_score)
        correct = correct_score < min(distractor_scores)

        # End context: promotes patterns to Tier 2 only on correct tasks
        if use_contextual:
            store_obj.end_context(context_id, success_metrics={"correct": correct})
        else:
            tiered.end_context(context_id, correct=correct)

        # Off-line cross-task recombination every 10 tasks
        if (run_idx + 1) % 10 == 0:
            for agent in agents:
                recombinator.consolidate(tiered, agent.agent_id)

        if correct:
            correct_count += 1
        rank_sum += rank

        if (run_idx + 1) % 50 == 0:
            pct = correct_count / (run_idx + 1) * 100
            t2_count = len(tiered.query_tier2_all())
            n_patterns = sum(len(agent.store.query(agent.agent_id)) for agent in agents)
            t1_count = n_patterns - t2_count
            print(
                f"  {run_idx + 1}/{len(eligible)} — accuracy: {pct:.1f}%, "
                f"tier1: {t1_count}, tier2_meta: {t2_count}",
                flush=True,
            )

    return correct_count, rank_sum


def main():
    import multiprocessing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count, ignored in --persistent mode)")
    parser.add_argument("--persistent", action="store_true",
                        help="Agents learn across tasks (sequential, shared store)")
    parser.add_argument("--no-contextual", action="store_true",
                        help="Use bare TieredStore instead of ContextualPatternStore (for comparison)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks evaluated (useful for quick testing)")
    args = parser.parse_args()

    print("Loading ARC dataset...", flush=True)
    all_tasks = load_tasks()

    eligible = [(i, t) for i, t in enumerate(all_tasks) if task_fits(t)]
    excluded = len(all_tasks) - len(eligible)

    if args.max_tasks is not None:
        eligible = eligible[:args.max_tasks]

    print(
        f"Tasks: {len(all_tasks)} total, {excluded} excluded "
        f"(grid > {MAX_GRID_DIM}x{MAX_GRID_DIM}), {len(eligible)} evaluated"
    )

    correct_count = 0
    rank_sum = 0

    if args.persistent:
        use_contextual = not args.no_contextual
        mode_label = "contextual ContextualPatternStore" if use_contextual else "bare TieredStore"
        print(f"Running (2-agent persistent, sequential, {mode_label})...", flush=True)
        correct_count, rank_sum = run_persistent(eligible, all_tasks, use_contextual=use_contextual)
    else:
        n_workers = args.workers or multiprocessing.cpu_count()
        print(f"Running (2-agent ensemble, {n_workers} workers)...", flush=True)
        work = [(task, task_idx) for task_idx, task in eligible]
        with multiprocessing.Pool(processes=n_workers,
                                  initializer=_init_worker,
                                  initargs=(all_tasks,)) as pool:
            for run_idx, (correct, rank) in enumerate(pool.imap_unordered(_eval_worker, work)):
                if correct:
                    correct_count += 1
                rank_sum += rank
                if (run_idx + 1) % 50 == 0:
                    pct = correct_count / (run_idx + 1) * 100
                    print(f"  {run_idx + 1}/{len(eligible)} tasks — accuracy so far: {pct:.1f}%", flush=True)

    n = len(eligible)
    accuracy = correct_count / n * 100 if n > 0 else 0.0
    mean_rank = rank_sum / n if n > 0 else 3.0
    vs_chance = accuracy - 20.0

    assert 0.0 <= accuracy <= 100.0
    assert 1.0 <= mean_rank <= 5.0

    print_results_table(
        title="Multi-Agent ARC Benchmark (2-agent ensemble, 5-way discrimination, fchollet/arc-agi train split)",
        cols=["Tasks Run", "Excluded", "Correct", "Accuracy", "Mean Rank", "vs Chance"],
        rows=[{
            "Tasks Run": str(n),
            "Excluded": str(excluded),
            "Correct": str(correct_count),
            "Accuracy": f"{accuracy:.1f}%",
            "Mean Rank": f"{mean_rank:.2f}",
            "vs Chance": f"{vs_chance:+.1f}%",
        }],
    )


if __name__ == "__main__":
    main()
