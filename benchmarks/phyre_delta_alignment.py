"""SP10: Delta Alignment — Structure-Preserving Cross-Domain Transfer.

Aligns the relational topology (ΔL2→ΔL3 mappings) across domains via
Procrustes rotation, rather than aligning raw feature distributions.

Three leave-one-domain-out rotations:
  Math + PhyRE  → ARC
  Math + ARC    → PhyRE
  PhyRE + ARC   → Math

For each rotation, reports:
  l2l3            — domain-native L2+L3 baseline (no cross-domain)
  cross_domain_l4 — SP9 globally trained L4GenerativeHead result (for comparison)
  delta_alignment — SP10 Procrustes-aligned delta matrix scoring
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

PAD_DIM = 14  # shared padded dimension for both L2 and L3

# SP9 cross_domain_l4 results (from phyre_cross_domain_l4.py) for comparison
_SP9_RESULTS = {
    ('math', 'phyre', 'arc'):   {'l2l3': 0.800, 'cross_domain_l4': 0.267},
    ('math', 'arc',   'phyre'): {'l2l3': 0.583, 'cross_domain_l4': 0.167},
    ('phyre', 'arc',  'math'):  {'l2l3': 1.000, 'cross_domain_l4': 0.222},
}


# --------------------------------------------------------------------------- #
#  Helper: zero-pad a vector to length `target_dim`
# --------------------------------------------------------------------------- #

def _pad(vec: np.ndarray, target_dim: int = PAD_DIM) -> np.ndarray:
    """Zero-pad *vec* to *target_dim* (or truncate if longer)."""
    v = np.asarray(vec, dtype=np.float64)
    if len(v) >= target_dim:
        return v[:target_dim]
    return np.pad(v, (0, target_dim - len(v)))


# --------------------------------------------------------------------------- #
#  Domain loaders and encoder factories (reused from SP9)
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


def _load_math_tasks(n_per_family: int) -> list[dict]:
    from benchmarks.structured_math import generate_tasks
    return generate_tasks(n_per_family=n_per_family, seed=42)


def _load_phyre_tasks(n_per_family: int) -> list[dict]:
    from benchmarks.phyre_sim import generate_family_tasks
    families = ['Projectile', 'Bounce', 'Slide', 'Collision']
    tasks = []
    for fam in families:
        tasks.extend(generate_family_tasks(fam, n_tasks=n_per_family, seed=42))
    return tasks


def _load_arc_tasks(n_per_family: int, rng: np.random.Generator) -> list[dict]:
    from benchmarks.multi_agent_arc import load_tasks, task_fits, N_DISTRACTORS

    all_tasks = [t for t in load_tasks() if task_fits(t)]
    if not all_tasks:
        return []

    chosen = all_tasks[:min(n_per_family, len(all_tasks))]
    remaining = all_tasks

    result = []
    for i, task in enumerate(chosen):
        test_pair = task["test"][0]
        test_input = test_pair["input"]
        test_output = test_pair["output"]

        other_indices = [j for j in range(len(remaining)) if j != i]
        if len(other_indices) < N_DISTRACTORS:
            continue
        chosen_dist = rng.choice(other_indices, size=N_DISTRACTORS, replace=False)
        distractor_outputs = [remaining[di]["train"][0]["output"] for di in chosen_dist]

        candidates = [test_output] + distractor_outputs

        train_pairs = [(p["input"], p["output"]) for p in task["train"]]
        result.append({
            "train": train_pairs,
            "test_input": test_input,
            "test_output": test_output,
            "candidates": candidates,
            "correct_idx": 0,
        })

    return result


def _load(domain: str, n_per_family: int, rng: np.random.Generator) -> list[dict]:
    if domain == 'math':
        return _load_math_tasks(n_per_family)
    elif domain == 'phyre':
        return _load_phyre_tasks(n_per_family)
    elif domain == 'arc':
        return _load_arc_tasks(n_per_family, rng)
    return []


def _correct_idx(task: dict, domain: str) -> int:
    if domain == 'phyre':
        return int(task['test']['correct_idx'])
    return int(task.get('correct_idx', 0))


# --------------------------------------------------------------------------- #
#  Phase 1: compute_delta_pairs
# --------------------------------------------------------------------------- #

def compute_delta_pairs(
    tasks: list[dict],
    domain: str,
    l2_enc,
    l3_enc,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute all N(N-1) pairwise (ΔL2, ΔL3) deltas from train pairs.

    Each train pair is encoded to (L2_padded, L3_padded).
    For all ordered pairs (i, j) where i != j:
        ΔL2_ij = L2_i - L2_j   (zero-padded to PAD_DIM)
        ΔL3_ij = L3_i - L3_j   (zero-padded to PAD_DIM)

    Returns list of (ΔL2, ΔL3) tuples.
    """
    # First encode all train pairs across all tasks
    encoded: list[tuple[np.ndarray, np.ndarray]] = []

    for task in tasks:
        train = task.get('train', [])
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

                l2_vecs = l2_enc.encode(obs_l2, epistemic=None)
                l3_vecs = l3_enc.encode(obs_l3, epistemic=None)

                if not l2_vecs or not l3_vecs:
                    continue

                l2_pad = _pad(l2_vecs[0])
                l3_pad = _pad(l3_vecs[0])
                encoded.append((l2_pad, l3_pad))

            except Exception:
                continue

    # Compute all N(N-1) ordered pairwise deltas
    delta_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(encoded)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dl2 = encoded[i][0] - encoded[j][0]
            dl3 = encoded[i][1] - encoded[j][1]
            delta_pairs.append((dl2, dl3))

    return delta_pairs


# --------------------------------------------------------------------------- #
#  Phase 1: fit_domain_matrix
# --------------------------------------------------------------------------- #

def fit_domain_matrix(
    delta_pairs: list[tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.01,
) -> np.ndarray:
    """Fit local mapping M_d ∈ R^{14×14} via ridge regression.

    Solves: M_d @ ΔL2 ≈ ΔL3
    Ridge formula: M = (Y^T X (X^T X + α I)^{-1})^T
    i.e., M = solve(X^T X + α I, X^T Y).T

    Args:
        delta_pairs: list of (ΔL2, ΔL3) tuples, each of shape (PAD_DIM,)
        alpha: ridge regularisation strength

    Returns:
        M_d of shape (PAD_DIM, PAD_DIM)
    """
    if not delta_pairs:
        return np.eye(PAD_DIM)

    X = np.array([dl2 for dl2, dl3 in delta_pairs], dtype=np.float64)  # (N, 14)
    Y = np.array([dl3 for dl2, dl3 in delta_pairs], dtype=np.float64)  # (N, 14)

    A = X.T @ X + alpha * np.eye(PAD_DIM)   # (14, 14)
    M = np.linalg.solve(A, X.T @ Y).T        # (14, 14)

    return M


# --------------------------------------------------------------------------- #
#  Phase 2: procrustes_align
# --------------------------------------------------------------------------- #

def procrustes_align(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Align M2 to M1 via Procrustes rotation and return shared matrix.

    Algorithm:
        SVD: M2 @ M1^T = U Σ V^T
        Det check: d = sign(det(V @ U^T))
        D = diag([1]*13 + [d])   # flip last col if det < 0
        R = V @ D @ U^T          # R ∈ SO(14)
        M_shared = (M1 + R^T @ M2) / 2

    Args:
        M1: domain matrix 1, shape (PAD_DIM, PAD_DIM)
        M2: domain matrix 2, shape (PAD_DIM, PAD_DIM)

    Returns:
        M_shared of shape (PAD_DIM, PAD_DIM)
    """
    U, s, Vt = np.linalg.svd(M2 @ M1.T)
    V = Vt.T

    d = np.sign(np.linalg.det(V @ U.T))
    if d == 0:
        d = 1.0  # handle degenerate case

    D = np.diag([1.0] * (PAD_DIM - 1) + [float(d)])
    R = V @ D @ U.T

    M_shared = (M1 + R.T @ M2) / 2.0
    return M_shared


# --------------------------------------------------------------------------- #
#  Phase 3: score_with_anchor
# --------------------------------------------------------------------------- #

def score_with_anchor(
    task: dict,
    domain: str,
    M_shared: np.ndarray,
    l2_enc,
    l3_enc,
) -> int:
    """Score candidates using centroid anchor + shared delta matrix.

    Algorithm:
        anchor_l2 = mean(L2 from train pairs)
        anchor_l3 = mean(L3 from train pairs)
        For each candidate:
            ΔL2 = L2_cand - anchor_l2   (padded)
            ΔL3_pred = M_shared @ ΔL2
            ΔL3_actual = L3_cand - anchor_l3   (padded)
            score = -||ΔL3_actual - ΔL3_pred||
        Return argmax(scores)

    Returns:
        Index of highest-scoring candidate.
    """
    train = task.get('train', [])

    # Encode train pairs for anchor
    train_l2 = []
    train_l3 = []
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
                train_l2.append(_pad(vecs2[0]))
            if vecs3:
                train_l3.append(_pad(vecs3[0]))
        except Exception:
            continue

    if not train_l2 or not train_l3:
        return 0

    anchor_l2 = np.mean(train_l2, axis=0)
    anchor_l3 = np.mean(train_l3, axis=0)

    # Get test input and candidates
    if domain == 'phyre':
        test_input = task['test']['init']
        candidates = task['test']['candidates']
    else:
        test_input = task.get('test_input')
        candidates = task.get('candidates', [])

    if not candidates:
        return 0

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
                scores.append(-np.inf)
                continue

            vecs2 = l2_enc.encode(obs_l2, epistemic=None)
            vecs3 = l3_enc.encode(obs_l3, epistemic=None)

            if not vecs2 or not vecs3:
                scores.append(-np.inf)
                continue

            l2_cand = _pad(vecs2[0])
            l3_cand = _pad(vecs3[0])

            dl2 = l2_cand - anchor_l2
            dl3_pred = M_shared @ dl2
            dl3_actual = l3_cand - anchor_l3

            score = -float(np.linalg.norm(dl3_actual - dl3_pred))
            scores.append(score)

        except Exception:
            scores.append(-np.inf)

    if not scores:
        return 0
    return int(np.argmax(scores))


# --------------------------------------------------------------------------- #
#  Run one leave-one-out rotation
# --------------------------------------------------------------------------- #

def run_rotation(
    train_domains: list[str],
    test_domain: str,
    n_per_family: int = 15,
    seed: int = 42,
) -> dict[str, float]:
    """Run one leave-one-out rotation.

    Returns dict with keys: l2l3, delta_alignment, n_test, n_train_pairs.
    """
    rng = np.random.default_rng(seed)

    # Load tasks
    train_tasks_by_domain = {d: _load(d, n_per_family, rng) for d in train_domains}
    test_tasks = _load(test_domain, n_per_family, rng)

    if not test_tasks:
        return {'l2l3': 0.0, 'delta_alignment': 0.0, 'n_test': 0, 'n_train_pairs': 0}

    # Phase 1: compute delta pairs and fit domain matrices for each training domain
    domain_matrices: dict[str, np.ndarray] = {}
    total_train_pairs = 0

    for domain in train_domains:
        l2_enc, l3_enc = _get_encoders(domain)
        tasks = train_tasks_by_domain[domain]
        delta_pairs = compute_delta_pairs(tasks, domain, l2_enc, l3_enc)
        total_train_pairs += len(delta_pairs)
        M_d = fit_domain_matrix(delta_pairs)
        domain_matrices[domain] = M_d

    # Phase 2: Procrustes alignment to get M_shared
    assert len(train_domains) == 2, "Expected exactly 2 training domains"
    d1, d2 = train_domains
    M_shared = procrustes_align(domain_matrices[d1], domain_matrices[d2])

    # Phase 3: Evaluate on test domain
    test_l2_enc, test_l3_enc = _get_encoders(test_domain)

    # Also run l2l3 baseline for comparison (reuse SP9 logic)
    from benchmarks.phyre_cross_domain_l4 import score_l2l3_baseline

    l2l3_correct = 0
    delta_correct = 0

    for task in test_tasks:
        correct = _correct_idx(task, test_domain)

        if test_domain == 'phyre':
            # Build flat task for l2l3 baseline
            flat_task = {
                'train': task['train'],
                'test_input': task['test']['init'],
                'candidates': task['test']['candidates'],
                'correct_idx': correct,
            }
        else:
            flat_task = task

        pred_l2l3 = score_l2l3_baseline(flat_task, test_domain, test_l2_enc, test_l3_enc)
        pred_delta = score_with_anchor(task, test_domain, M_shared, test_l2_enc, test_l3_enc)

        if pred_l2l3 == correct:
            l2l3_correct += 1
        if pred_delta == correct:
            delta_correct += 1

    n = len(test_tasks)
    return {
        'l2l3': l2l3_correct / n if n else 0.0,
        'delta_alignment': delta_correct / n if n else 0.0,
        'n_test': n,
        'n_train_pairs': total_train_pairs,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    rotations = [
        (['math', 'phyre'], 'arc'),
        (['math', 'arc'],   'phyre'),
        (['phyre', 'arc'],  'math'),
    ]

    print("SP10 Delta Alignment Benchmark")
    print(
        f"{'Train domains':<20} {'Test':>6}  {'l2l3':>6}  {'cross_domain_l4':>15}  {'delta_alignment':>15}"
    )
    print("-" * 72)

    summary: list[dict] = []
    for train_domains, test_domain in rotations:
        r = run_rotation(train_domains, test_domain, n_per_family=15)
        label = ' + '.join(d.capitalize() for d in train_domains)

        # Look up SP9 cross_domain_l4 result
        key = tuple(train_domains + [test_domain])
        sp9 = _SP9_RESULTS.get(key, {})
        cdl4 = sp9.get('cross_domain_l4', float('nan'))

        print(
            f"{label:<20} {test_domain:>6}  {r['l2l3']:>6.3f}  {cdl4:>15.3f}  {r['delta_alignment']:>15.3f}"
        )
        summary.append({
            'train': label,
            'test': test_domain,
            'l2l3': r['l2l3'],
            'cross_domain_l4': cdl4,
            'delta_alignment': r['delta_alignment'],
        })

    # Verdict
    beats_l2l3 = sum(1 for s in summary if s['delta_alignment'] > s['l2l3'])
    beats_cdl4 = sum(
        1 for s in summary
        if not np.isnan(s['cross_domain_l4']) and s['delta_alignment'] > s['cross_domain_l4']
    )
    print()
    if beats_l2l3 >= 2:
        verdict = "STRONG: delta_alignment > l2l3 on >= 2/3 rotations"
    elif beats_cdl4 >= 1:
        verdict = f"PARTIAL: delta_alignment > cross_domain_l4 on {beats_cdl4}/3 rotations (but < l2l3)"
    else:
        verdict = "NEGATIVE: delta_alignment does not exceed cross_domain_l4"
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
