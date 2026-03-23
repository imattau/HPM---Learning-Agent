"""
Structured ARC Benchmark
========================
Three-level HPM benchmark: each level receives a genuinely different abstraction.

  L1 (feature_dim=64): pixel delta -- sensory regularities
  L2 (feature_dim=9):  per-object anatomy -- object-level witnesses
  L3 (feature_dim=14): relational summary -- transformation families

Four baselines compared in one run:
  flat         -- 2-agent pixel delta, partitioned training (matches multi_agent_arc.py)
  l1_only      -- full structured training, scored via L1 only
  l2_only      -- full structured training, scored via L2 mean only
  full         -- full structured training, L1 + mean(L2) + L3

Run:
    python benchmarks/structured_arc.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.multi_agent_arc import load_tasks, task_fits, TRAIN_REPS, N_DISTRACTORS, ensemble_score
from benchmarks.multi_agent_common import make_orchestrator, print_results_table
from benchmarks.arc_encoders import (
    ArcL1Encoder, ArcL2Encoder, ArcL3Encoder, _encode_grid_flat, _L1_PROJ,
)
from hpm.agents.structured import StructuredOrchestrator


def _make_structured_orch():
    """Build L1(2-agent) + L2(2-agent) + L3(1-agent) StructuredOrchestrator."""
    l1_orch, l1_agents, _ = make_orchestrator(
        n_agents=2, feature_dim=64, agent_ids=["l1_0", "l1_1"],
        pattern_types=["gaussian", "gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    l2_orch, l2_agents, _ = make_orchestrator(
        n_agents=2, feature_dim=9, agent_ids=["l2_0", "l2_1"],
        pattern_types=["gaussian", "gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    l3_orch, l3_agents, _ = make_orchestrator(
        n_agents=1, feature_dim=14, agent_ids=["l3_0"],
        pattern_types=["gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    enc1, enc2, enc3 = ArcL1Encoder(), ArcL2Encoder(), ArcL3Encoder()
    orch = StructuredOrchestrator(
        encoders=[enc1, enc2, enc3],
        orches=[l1_orch, l2_orch, l3_orch],
        agents=[l1_agents, l2_agents, l3_agents],
        level_Ks=[1, 1, 3],  # L2 always fires; L3 every 3 steps
    )
    return orch, l1_agents, l2_agents, l3_agents


def _make_flat_orch():
    """2-agent flat baseline (matches multi_agent_arc.py)."""
    orch, agents, _ = make_orchestrator(
        n_agents=2, feature_dim=64, agent_ids=["flat_0", "flat_1"],
        pattern_types=["gaussian", "gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    return orch, agents


def _score_structured(l1_agents, l2_agents, l3_agents, obs, l1_ep, l2_ep):
    """Compute L1, L2 (mean), and L3 scores for one candidate observation."""
    l1_enc = ArcL1Encoder()
    l2_enc = ArcL2Encoder()
    l3_enc = ArcL3Encoder()

    l1_vec = l1_enc.encode(obs, epistemic=None)[0]
    l2_vecs = l2_enc.encode(obs, epistemic=l1_ep)
    l3_vec = l3_enc.encode(obs, epistemic=l2_ep)[0]

    l1_score = ensemble_score(l1_agents, l1_vec)
    l2_score = (
        float(np.mean([ensemble_score(l2_agents, v) for v in l2_vecs]))
        if l2_vecs else 0.0
    )
    l3_score = ensemble_score(l3_agents, l3_vec)
    return l1_score, l2_score, l3_score


def run(max_tasks: int | None = None) -> dict:
    tasks = load_tasks()
    tasks = [t for t in tasks if task_fits(t)][:400]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    rng = np.random.default_rng(42)

    flat_correct = l1_correct = l2_correct = full_correct = 0
    n_evaluated = 0

    for i, task in enumerate(tasks):
        test_pair = task["test"][0]
        test_input = test_pair["input"]
        correct_obs = (test_input, test_pair["output"])

        distractor_idxs = [j for j in range(len(tasks)) if j != i]
        chosen = rng.choice(distractor_idxs, size=N_DISTRACTORS, replace=False)
        distractor_obs = [
            (test_input, tasks[di]["train"][0]["output"]) for di in chosen
        ]
        all_obs = [correct_obs] + distractor_obs

        # Partitioned training pairs
        train_pairs = task["train"]
        pairs_a = train_pairs[0::2] or train_pairs
        pairs_b = train_pairs[1::2] or train_pairs
        n_pairs = max(len(pairs_a), len(pairs_b))

        # --- Structured ---
        orch, l1_agents, l2_agents, l3_agents = _make_structured_orch()
        l1_enc = ArcL1Encoder()
        l1_ids = [a.agent_id for a in l1_agents]

        for _ in range(TRAIN_REPS):
            for k in range(n_pairs):
                obs_a = (pairs_a[k % len(pairs_a)]["input"], pairs_a[k % len(pairs_a)]["output"])
                obs_b = (pairs_b[k % len(pairs_b)]["input"], pairs_b[k % len(pairs_b)]["output"])
                l1_obs_dict = {
                    l1_ids[0]: l1_enc.encode(obs_a, epistemic=None)[0],
                    l1_ids[1]: l1_enc.encode(obs_b, epistemic=None)[0],
                }
                orch.step(obs_a, l1_obs_dict=l1_obs_dict)

        # Extract end-of-training epistemic state
        l1_ep = orch._epistemic[0] or (0.0, 0.0)
        l2_ep = orch._epistemic[1] or (0.0, 0.0)

        # Score all candidates
        all_scores = [
            _score_structured(l1_agents, l2_agents, l3_agents, obs, l1_ep, l2_ep)
            for obs in all_obs
        ]
        l1_scores = [s[0] for s in all_scores]
        l2_scores = [s[1] for s in all_scores]
        combined = [s[0] + s[1] + s[2] for s in all_scores]

        if l1_scores[0] == min(l1_scores):
            l1_correct += 1
        if l2_scores[0] == min(l2_scores):
            l2_correct += 1
        if combined[0] == min(combined):
            full_correct += 1

        # --- Flat baseline ---
        flat_orch, flat_agents = _make_flat_orch()
        flat_ids = [a.agent_id for a in flat_agents]
        for _ in range(TRAIN_REPS):
            for k in range(n_pairs):
                obs_a = (pairs_a[k % len(pairs_a)]["input"], pairs_a[k % len(pairs_a)]["output"])
                obs_b = (pairs_b[k % len(pairs_b)]["input"], pairs_b[k % len(pairs_b)]["output"])
                flat_l1_dict = {
                    flat_ids[0]: l1_enc.encode(obs_a, epistemic=None)[0],
                    flat_ids[1]: l1_enc.encode(obs_b, epistemic=None)[0],
                }
                flat_orch.step(flat_l1_dict)

        flat_vecs = [l1_enc.encode(obs, epistemic=None)[0] for obs in all_obs]
        flat_scores = [ensemble_score(flat_agents, v) for v in flat_vecs]
        if flat_scores[0] == min(flat_scores):
            flat_correct += 1

        n_evaluated += 1

    chance = 1.0 / (N_DISTRACTORS + 1)
    return {
        "n_tasks": n_evaluated,
        "flat_acc": flat_correct / n_evaluated if n_evaluated else 0.0,
        "l1_acc": l1_correct / n_evaluated if n_evaluated else 0.0,
        "l2_acc": l2_correct / n_evaluated if n_evaluated else 0.0,
        "full_acc": full_correct / n_evaluated if n_evaluated else 0.0,
        "chance": chance,
    }


def main():
    print("Running Structured ARC Benchmark (L1=pixel, L2=object, L3=relational)...")
    m = run()
    chance = m["chance"]
    print_results_table(
        title=f"Structured ARC Benchmark ({m['n_tasks']} tasks, per-task reset)",
        cols=["Setup", "Accuracy", "vs Chance"],
        rows=[
            {"Setup": "Flat (2-agent, pixel only)", "Accuracy": f"{m['flat_acc']:.1%}", "vs Chance": f"{m['flat_acc']-chance:+.1%}"},
            {"Setup": "L1 only (structured train)", "Accuracy": f"{m['l1_acc']:.1%}", "vs Chance": f"{m['l1_acc']-chance:+.1%}"},
            {"Setup": "L2 only (object anatomy)",  "Accuracy": f"{m['l2_acc']:.1%}", "vs Chance": f"{m['l2_acc']-chance:+.1%}"},
            {"Setup": "Full (L1+L2+L3 combined)",  "Accuracy": f"{m['full_acc']:.1%}", "vs Chance": f"{m['full_acc']-chance:+.1%}"},
        ],
    )


if __name__ == "__main__":
    main()
