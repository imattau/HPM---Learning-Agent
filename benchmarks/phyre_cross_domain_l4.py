"""SP9: Cross-Domain L4 Generalisation Benchmark.

Leave-one-domain-out: train a single L4GenerativeHead on two domains,
evaluate on the third. Zero-pads all L2 to 14-dim and all L3 to 14-dim
so a single head covers all three domains.

Three rotations:
  Math + PhyRE  → ARC
  Math + ARC    → PhyRE
  PhyRE + ARC   → Math

For each rotation, reports:
  l2l3            — domain-native L2+L3 baseline (no L4)
  cross_domain_l4 — globally trained L4GenerativeHead (cross-domain transfer)
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from hpm.agents.l4_generative import L4GenerativeHead

# --------------------------------------------------------------------------- #
#  Shared constants
# --------------------------------------------------------------------------- #

PAD_DIM = 14   # shared padded dimension for both L2 and L3

# Number of tasks per family/domain for the benchmark.
# Set to a small value here; override via CLI --n-tasks.
_DEFAULT_N_TASKS = 15


# --------------------------------------------------------------------------- #
#  Helper: zero-pad a vector to length `target_dim`
# --------------------------------------------------------------------------- #

def _pad(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad *vec* to *target_dim* (or truncate if longer)."""
    v = np.asarray(vec, dtype=np.float32)
    if len(v) >= target_dim:
        return v[:target_dim]
    return np.pad(v, (0, target_dim - len(v)))


# --------------------------------------------------------------------------- #
#  Domain-specific task generators
# --------------------------------------------------------------------------- #

def _load_math_tasks(n_tasks: int) -> list[dict]:
    """Return math tasks (structured_math format)."""
    from benchmarks.structured_math import generate_tasks
    n_per_family = max(1, n_tasks // 5)
    return generate_tasks(n_per_family=n_per_family, seed=42)


def _load_phyre_tasks(n_tasks: int) -> list[dict]:
    """Return PhyRE tasks across 4 families."""
    from benchmarks.phyre_sim import generate_family_tasks
    families = ['Projectile', 'Bounce', 'Slide', 'Collision']
    n_per_family = max(1, n_tasks // len(families))
    tasks = []
    for fam in families:
        tasks.extend(generate_family_tasks(fam, n_tasks=n_per_family, seed=42))
    return tasks


def _load_arc_tasks(n_tasks: int, rng: np.random.Generator) -> list[dict]:
    """Return ARC tasks with pre-sampled distractor candidates.

    Returns tasks in a unified format:
      train:        list of (grid_in, grid_out)
      test_input:   grid
      test_output:  grid
      candidates:   list of grids (correct at index 0)
      correct_idx:  0
    """
    from benchmarks.multi_agent_arc import load_tasks, task_fits, N_DISTRACTORS

    all_tasks = [t for t in load_tasks() if task_fits(t)]
    if len(all_tasks) == 0:
        return []

    chosen = all_tasks[: min(n_tasks, len(all_tasks))]
    remaining = all_tasks  # pool for distractors

    result = []
    for i, task in enumerate(chosen):
        test_pair = task["test"][0]
        test_input = test_pair["input"]
        test_output = test_pair["output"]

        # Build distractor pool (outputs from other tasks)
        other_indices = [j for j in range(len(remaining)) if j != i]
        if len(other_indices) < N_DISTRACTORS:
            continue
        chosen_dist = rng.choice(other_indices, size=N_DISTRACTORS, replace=False)
        distractor_outputs = [remaining[di]["train"][0]["output"] for di in chosen_dist]

        candidates = [test_output] + distractor_outputs  # correct at idx 0

        train_pairs = [
            (p["input"], p["output"]) for p in task["train"]
        ]
        result.append({
            "train": train_pairs,
            "test_input": test_input,
            "test_output": test_output,
            "candidates": candidates,
            "correct_idx": 0,
        })

    return result


# --------------------------------------------------------------------------- #
#  Encoder factories
# --------------------------------------------------------------------------- #

def _get_encoders(domain: str):
    """Return (l2_enc, l3_enc) for the given domain."""
    if domain == 'math':
        from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder
        return MathL2Encoder(), MathL3Encoder()
    elif domain == 'phyre':
        from hpm.encoders.phyre_encoders import PhyreL2Encoder, PhyreL3Encoder
        return PhyreL2Encoder(), PhyreL3Encoder()
    elif domain == 'arc':
        from benchmarks.arc_encoders import ArcL2Encoder, ArcL3Encoder
        return ArcL2Encoder(), ArcL3Encoder()
    else:
        raise ValueError(f"Unknown domain: {domain}")


# --------------------------------------------------------------------------- #
#  Core: encode domain pairs for L4 training
# --------------------------------------------------------------------------- #

def encode_domain_pairs(
    tasks: list[dict],
    domain: str,
    l2_enc,
    l3_enc,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Encode all train pairs for *domain*, zero-pad to (PAD_DIM, PAD_DIM).

    Returns list of (l2_padded, l3_padded) tuples, one per valid train pair.
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for task in tasks:
        train = task.get('train', [])
        for pair in train:
            try:
                if domain == 'math':
                    obs = (pair[0], pair[1])
                    l2_vecs = l2_enc.encode(obs, epistemic=None)
                    l3_vecs = l3_enc.encode(obs, epistemic=None)

                elif domain == 'phyre':
                    obs_l2 = (pair['init'], pair['final'])
                    obs_l3 = (pair['init'], pair['final'], 0)
                    l2_vecs = l2_enc.encode(obs_l2, epistemic=None)
                    l3_vecs = l3_enc.encode(obs_l3, epistemic=None)

                elif domain == 'arc':
                    obs = (pair[0], pair[1])
                    l2_vecs = l2_enc.encode(obs, epistemic=None)
                    l3_vecs = l3_enc.encode(obs, epistemic=None)

                else:
                    continue

                if not l2_vecs or not l3_vecs:
                    continue

                l2_pad = _pad(l2_vecs[0], PAD_DIM)
                l3_pad = _pad(l3_vecs[0], PAD_DIM)
                pairs.append((l2_pad, l3_pad))

            except Exception:
                continue

    return pairs


# --------------------------------------------------------------------------- #
#  Core: fit global L4
# --------------------------------------------------------------------------- #

def fit_cross_domain_l4(
    all_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> L4GenerativeHead:
    """Accumulate all (l2_padded, l3_padded) pairs and fit global head."""
    head = L4GenerativeHead(feature_dim_in=PAD_DIM, feature_dim_out=PAD_DIM)
    for l2, l3 in all_pairs:
        head.accumulate(l2, l3)
    head.fit()
    return head


# --------------------------------------------------------------------------- #
#  Scoring: cross-domain L4
# --------------------------------------------------------------------------- #

def score_cross_domain(
    task: dict,
    domain: str,
    global_l4: L4GenerativeHead,
    l2_enc,
    l3_enc,
) -> int:
    """Score candidates using global L4. Returns predicted candidate index."""
    test_input = task.get('test_input') or task.get('test', {}).get('init')
    candidates = task.get('candidates', [])

    if not candidates:
        return 0

    scores = []
    for cand in candidates:
        try:
            if domain == 'math':
                obs_l2 = (test_input, cand)
                obs_l3 = (test_input, cand)
                l2_vecs = l2_enc.encode(obs_l2, epistemic=None)
                l3_vecs = l3_enc.encode(obs_l3, epistemic=None)

            elif domain == 'phyre':
                cand_final = cand['final']
                obs_l2 = (test_input, cand_final)
                obs_l3 = (test_input, cand_final, 0)
                l2_vecs = l2_enc.encode(obs_l2, epistemic=None)
                l3_vecs = l3_enc.encode(obs_l3, epistemic=None)

            elif domain == 'arc':
                obs_l2 = (test_input, cand)
                obs_l3 = (test_input, cand)
                l2_vecs = l2_enc.encode(obs_l2, epistemic=None)
                l3_vecs = l3_enc.encode(obs_l3, epistemic=None)

            else:
                scores.append(0.0)
                continue

            if not l2_vecs or not l3_vecs:
                scores.append(0.0)
                continue

            l2_pad = _pad(l2_vecs[0], PAD_DIM)
            l3_pad = _pad(l3_vecs[0], PAD_DIM)

            pred_l3 = global_l4.predict(l2_pad)
            if pred_l3 is None:
                scores.append(0.0)
            else:
                error = float(np.linalg.norm(pred_l3 - l3_pad))
                scores.append(-error)   # higher = better

        except Exception:
            scores.append(0.0)

    return int(np.argmax(scores))


# --------------------------------------------------------------------------- #
#  Scoring: domain-native L2+L3 baseline
# --------------------------------------------------------------------------- #

def score_l2l3_baseline(
    task: dict,
    domain: str,
    l2_enc,
    l3_enc,
) -> int:
    """Score candidates using domain-native L2+L3 prototype matching.

    Computes mean L2 and L3 prototype from train pairs, picks the candidate
    with lowest combined L2+L3 distance (NLL proxy).
    """
    train = task.get('train', [])
    test_input = task.get('test_input') or task.get('test', {}).get('init')
    candidates = task.get('candidates', [])

    if not candidates:
        return 0

    # Build train prototypes
    train_l2_vecs = []
    train_l3_vecs = []
    for pair in train:
        try:
            if domain == 'math':
                obs_l2 = (pair[0], pair[1])
                obs_l3 = (pair[0], pair[1])
            elif domain == 'phyre':
                obs_l2 = (pair['init'], pair['final'])
                obs_l3 = (pair['init'], pair['final'], 0)
            elif domain == 'arc':
                obs_l2 = (pair[0], pair[1])
                obs_l3 = (pair[0], pair[1])
            else:
                continue

            vecs2 = l2_enc.encode(obs_l2, epistemic=None)
            vecs3 = l3_enc.encode(obs_l3, epistemic=None)
            if vecs2:
                train_l2_vecs.append(vecs2[0])
            if vecs3:
                train_l3_vecs.append(vecs3[0])
        except Exception:
            continue

    if not train_l2_vecs or not train_l3_vecs:
        return 0

    proto_l2 = np.mean(train_l2_vecs, axis=0)
    proto_l3 = np.mean(train_l3_vecs, axis=0)

    scores = []
    for cand in candidates:
        try:
            if domain == 'math':
                obs_l2 = (test_input, cand)
                obs_l3 = (test_input, cand)
            elif domain == 'phyre':
                cand_final = cand['final']
                obs_l2 = (test_input, cand_final)
                obs_l3 = (test_input, cand_final, 0)
            elif domain == 'arc':
                obs_l2 = (test_input, cand)
                obs_l3 = (test_input, cand)
            else:
                scores.append(float('inf'))
                continue

            vecs2 = l2_enc.encode(obs_l2, epistemic=None)
            vecs3 = l3_enc.encode(obs_l3, epistemic=None)

            if not vecs2 or not vecs3:
                scores.append(float('inf'))
                continue

            d2 = float(np.sum((vecs2[0] - proto_l2) ** 2))
            d3 = float(np.sum((vecs3[0] - proto_l3) ** 2))
            scores.append(d2 + d3)   # lower = better (NLL proxy)

        except Exception:
            scores.append(float('inf'))

    return int(np.argmin(scores))


# --------------------------------------------------------------------------- #
#  Correct-index extraction
# --------------------------------------------------------------------------- #

def _correct_idx(task: dict, domain: str) -> int:
    """Return the correct candidate index for a task."""
    if domain == 'phyre':
        return int(task['test']['correct_idx'])
    elif domain in ('math', 'arc'):
        return int(task.get('correct_idx', 0))
    return 0


# --------------------------------------------------------------------------- #
#  Run one leave-one-out rotation
# --------------------------------------------------------------------------- #

def run_rotation(
    train_domains: list[str],
    test_domain: str,
    n_per_domain: int = _DEFAULT_N_TASKS,
    seed: int = 42,
) -> dict[str, float]:
    """Run one leave-one-out rotation. Returns {'l2l3': float, 'cross_domain_l4': float}."""
    rng = np.random.default_rng(seed)

    # ---- Load tasks ----
    def _load(domain: str) -> list[dict]:
        if domain == 'math':
            return _load_math_tasks(n_per_domain)
        elif domain == 'phyre':
            return _load_phyre_tasks(n_per_domain)
        elif domain == 'arc':
            return _load_arc_tasks(n_per_domain, rng)
        return []

    train_tasks_by_domain = {d: _load(d) for d in train_domains}
    test_tasks = _load(test_domain)

    if not test_tasks:
        return {'l2l3': 0.0, 'cross_domain_l4': 0.0}

    # ---- Fit global L4 on train domains ----
    all_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for domain in train_domains:
        l2_enc, l3_enc = _get_encoders(domain)
        pairs = encode_domain_pairs(train_tasks_by_domain[domain], domain, l2_enc, l3_enc)
        all_pairs.extend(pairs)

    global_l4 = fit_cross_domain_l4(all_pairs)

    # ---- Evaluate on test domain ----
    test_l2_enc, test_l3_enc = _get_encoders(test_domain)

    l2l3_correct = 0
    cdl4_correct = 0

    for task in test_tasks:
        correct = _correct_idx(task, test_domain)

        # PhyRE tasks: restructure for scoring functions
        if test_domain == 'phyre':
            phyre_task = task
            # Build a scoring-compatible view
            test_init = phyre_task['test']['init']
            raw_candidates = phyre_task['test']['candidates']

            # score_l2l3_baseline and score_cross_domain expect task dict with
            # test_input + candidates (flat format). Build it:
            flat_task = {
                'train': phyre_task['train'],
                'test_input': test_init,
                'candidates': raw_candidates,  # each is {"action":..., "final":..., ...}
                'correct_idx': correct,
            }
        else:
            flat_task = task

        pred_l2l3 = score_l2l3_baseline(flat_task, test_domain, test_l2_enc, test_l3_enc)
        pred_cdl4 = score_cross_domain(flat_task, test_domain, global_l4, test_l2_enc, test_l3_enc)

        if pred_l2l3 == correct:
            l2l3_correct += 1
        if pred_cdl4 == correct:
            cdl4_correct += 1

    n = len(test_tasks)
    return {
        'l2l3': l2l3_correct / n if n else 0.0,
        'cross_domain_l4': cdl4_correct / n if n else 0.0,
        'n_test': n,
        'n_train_pairs': len(all_pairs),
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(n_tasks: int = _DEFAULT_N_TASKS) -> None:
    rotations = [
        (['math', 'phyre'], 'arc'),
        (['math', 'arc'],   'phyre'),
        (['phyre', 'arc'],  'math'),
    ]

    print("SP9 Cross-Domain L4 Benchmark")
    print(f"{'Train domains':<22} {'Test':>6}  {'n_train_pairs':>13}  {'n_test':>6}  {'l2l3':>6}  {'cross_domain_l4':>15}")
    print("-" * 80)

    summary: list[dict] = []
    for train_domains, test_domain in rotations:
        r = run_rotation(train_domains, test_domain, n_per_domain=n_tasks)
        label = '+'.join(train_domains)
        print(
            f"{label:<22} {test_domain:>6}  {r['n_train_pairs']:>13}  {r['n_test']:>6}  "
            f"{r['l2l3']:>6.3f}  {r['cross_domain_l4']:>15.3f}"
        )
        summary.append({'train': label, 'test': test_domain, **r})

    # Verdict
    beats = sum(1 for s in summary if s['cross_domain_l4'] > s['l2l3'])
    print()
    if beats >= 2:
        verdict = "STRONG: cross_domain_l4 > l2l3 on >= 2/3 rotations"
    elif beats == 1:
        verdict = "PARTIAL: cross_domain_l4 > l2l3 on 1/3 rotations"
    else:
        verdict = "NEGATIVE: cross_domain_l4 does not exceed l2l3"
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SP9 Cross-Domain L4 Benchmark")
    parser.add_argument('--n-tasks', type=int, default=_DEFAULT_N_TASKS,
                        help='Number of tasks per domain (default: %(default)s)')
    args = parser.parse_args()
    main(n_tasks=args.n_tasks)
