"""SP15: Generalized Cross-Domain Alignment.

Extends SP10 Procrustes alignment to 6 domains including symbolic "Boss Fights":
  - math, phyre, arc, ds1000, chem, linguistic

This script provides multi-domain loaders and a generalized alignment algorithm.
"""
from __future__ import annotations
import sys
import os
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))

# Shared padded dimension for all cross-domain experiments
PAD_DIM = 32

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _pad(vec: np.ndarray, target_dim: int = PAD_DIM) -> np.ndarray:
    """Zero-pad *vec* to *target_dim* (or truncate if longer)."""
    v = np.asarray(vec, dtype=np.float64)
    if v.ndim > 1:
        v = v.flatten()
    if len(v) >= target_dim:
        return v[:target_dim]
    return np.pad(v, (0, target_dim - len(v)))

def _encode_vec(encoder, pair: tuple, epistemic=None) -> np.ndarray:
    """Helper to encode a pair into a mean vector."""
    vecs = encoder.encode(pair, epistemic=epistemic)
    return np.mean(vecs, axis=0) if vecs else np.zeros(encoder.feature_dim)

# --------------------------------------------------------------------------- #
#  Domain Encoders Factory
# --------------------------------------------------------------------------- #

def get_encoders(domain: str):
    """Return (l2_enc, l3_enc) for any of the 6 HPM domains."""
    if domain == 'math':
        from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder
        return MathL2Encoder(), MathL3Encoder()
    elif domain == 'phyre':
        from hpm.encoders.phyre_encoders import PhyreL2Encoder, PhyreL3Encoder
        return PhyreL2Encoder(), PhyreL3Encoder()
    elif domain == 'arc':
        from benchmarks.arc_encoders import ArcL2Encoder, ArcL3Encoder
        return ArcL2Encoder(), ArcL3Encoder()
    elif domain == 'ds1000':
        from benchmarks.ds1000_encoders import DS1000L2Encoder, DS1000L3Encoder
        return DS1000L2Encoder(), DS1000L3Encoder()
    elif domain == 'chem':
        from benchmarks.chem_logic_encoders import ChemLogicL2Encoder, ChemLogicL3Encoder
        return ChemLogicL2Encoder(), ChemLogicL3Encoder()
    elif domain == 'linguistic':
        from benchmarks.linguistic_encoders import LinguisticL2Encoder, LinguisticL3Encoder
        return LinguisticL2Encoder(), LinguisticL3Encoder()
    else:
        raise ValueError(f"Unknown domain: {domain}")

# --------------------------------------------------------------------------- #
#  Domain Loaders
# --------------------------------------------------------------------------- #

def _load_math(n: int) -> list[dict]:
    from benchmarks.structured_math import generate_tasks
    return generate_tasks(n_per_family=max(1, n // 4), seed=42)

def _load_phyre(n: int) -> list[dict]:
    from benchmarks.phyre_sim import generate_family_tasks
    tasks = []
    for fam in ['Projectile', 'Bounce', 'Slide', 'Collision']:
        tasks.extend(generate_family_tasks(fam, n_tasks=max(1, n // 4), seed=42))
    return tasks

def _load_arc(n: int, rng: np.random.Generator) -> list[dict]:
    from benchmarks.multi_agent_arc import load_tasks, task_fits, N_DISTRACTORS
    all_tasks = [t for t in load_tasks() if task_fits(t)]
    chosen = all_tasks[:min(n, len(all_tasks))]
    result = []
    for i, task in enumerate(chosen):
        test_pair = task["test"][0]
        train_pairs = [(p["input"], p["output"]) for p in task["train"]]
        result.append({
            "train": train_pairs,
            "test_input": test_pair["input"],
            "test_output": test_pair["output"],
            "candidates": [test_pair["output"]] + [all_tasks[(i+j)%len(all_tasks)]["train"][0]["output"] for j in range(1, 5)],
            "correct_idx": 0,
        })
    return result

def _load_ds1000(n: int) -> list[dict]:
    from benchmarks.ds1000_sim import generate_ds1000_tasks
    return generate_ds1000_tasks(n_per_library=max(1, n // 7), seed=42)

def _load_chem(n: int) -> list[dict]:
    from benchmarks.chem_logic_sim import generate_chem_tasks
    return generate_chem_tasks(n_tasks=n, seed=42)

def _load_linguistic(n: int) -> list[dict]:
    from benchmarks.linguistic_sim import generate_register_tasks
    return generate_register_tasks(n_train=n, seed=42)

def load_domain_tasks(domain: str, n: int, rng: np.random.Generator) -> list[dict]:
    """Unified loader for all domains."""
    if domain == 'math': return _load_math(n)
    if domain == 'phyre': return _load_phyre(n)
    if domain == 'arc': return _load_arc(n, rng)
    if domain == 'ds1000': return _load_ds1000(n)
    if domain == 'chem': return _load_chem(n)
    if domain == 'linguistic': return _load_linguistic(n)
    return []

def get_obs_for_domain(domain: str, pair: tuple | dict):
    """Normalize (input, output) observation structure across domains."""
    if domain == 'phyre':
        if isinstance(pair, dict):
            return (pair['init'], pair['final'], 0)
        return pair
    return pair

# --------------------------------------------------------------------------- #
#  Phase 1: compute_delta_pairs & fit_domain_matrix
# --------------------------------------------------------------------------- #

def compute_delta_pairs(tasks: list[dict], domain: str, l2_enc, l3_enc) -> list[tuple[np.ndarray, np.ndarray]]:
    encoded: list[tuple[np.ndarray, np.ndarray]] = []
    for task in tasks:
        train = task.get('train', [])
        for pair in train:
            try:
                l2_vecs = l2_enc.encode(pair, epistemic=None)
                l3_vecs = l3_enc.encode(get_obs_for_domain(domain, pair), epistemic=None)
                if not l2_vecs or not l3_vecs: continue
                encoded.append((_pad(l2_vecs[0]), _pad(l3_vecs[0])))
            except: continue
    delta_pairs = []
    n = len(encoded)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            delta_pairs.append((encoded[i][0] - encoded[j][0], encoded[i][1] - encoded[j][1]))
    return delta_pairs

def fit_domain_matrix(delta_pairs: list[tuple[np.ndarray, np.ndarray]], alpha: float = 0.01) -> np.ndarray:
    if not delta_pairs: return np.eye(PAD_DIM)
    X = np.array([dl2 for dl2, dl3 in delta_pairs])
    Y = np.array([dl3 for dl2, dl3 in delta_pairs])
    A = X.T @ X + alpha * np.eye(PAD_DIM)
    return np.linalg.solve(A, X.T @ Y).T

# --------------------------------------------------------------------------- #
#  Phase 2: align_multiple_domains
# --------------------------------------------------------------------------- #

def procrustes_rotation(M_source: np.ndarray, M_target: np.ndarray) -> np.ndarray:
    U, s, Vt = np.linalg.svd(M_source @ M_target.T)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    if d == 0: d = 1.0
    D = np.diag([1.0] * (PAD_DIM - 1) + [float(d)])
    return V @ D @ U.T

def align_multiple_domains(matrices: list[np.ndarray], ref_idx: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    M_target = matrices[ref_idx]
    rotations, aligned = [], []
    for M in matrices:
        R = procrustes_rotation(M, M_target)
        rotations.append(R)
        aligned.append(R @ M)
    return np.mean(aligned, axis=0), rotations

# --------------------------------------------------------------------------- #
#  Phase 3: scoring & run_transfer_experiment
# --------------------------------------------------------------------------- #

def score_l2l3_baseline(task: dict, domain: str, l2_enc, l3_enc) -> int:
    train = task.get('train', [])
    train_l2, train_l3 = [], []
    for pair in train:
        try:
            v2 = l2_enc.encode(pair, epistemic=None)
            v3 = l3_enc.encode(get_obs_for_domain(domain, pair), epistemic=None)
            if v2: train_l2.append(_pad(v2[0]))
            if v3: train_l3.append(_pad(v3[0]))
        except: continue
    if not train_l2 or not train_l3: return 0
    a2, a3 = np.mean(train_l2, axis=0), np.mean(train_l3, axis=0)
    test_in = task.get('test_input') or (task['test']['init'] if domain == 'phyre' else None)
    candidates = task.get('candidates') or (task['test']['candidates'] if domain == 'phyre' else [])
    scores = []
    for cand in candidates:
        try:
            cf = cand['final'] if domain == 'phyre' else cand
            v2 = l2_enc.encode((test_in, cf), epistemic=None)
            v3 = l3_enc.encode(get_obs_for_domain(domain, (test_in, cf)), epistemic=None)
            if not v2 or not v3: scores.append(-1e9); continue
            scores.append(-float(np.linalg.norm(_pad(v2[0]) - a2)) - float(np.linalg.norm(_pad(v3[0]) - a3)))
        except: scores.append(-1e9)
    return int(np.argmax(scores))

def score_with_anchor(task: dict, domain: str, M_shared: np.ndarray, l2_enc, l3_enc) -> int:
    train = task.get('train', [])
    train_l2, train_l3 = [], []
    for pair in train:
        try:
            v2 = l2_enc.encode(pair, epistemic=None)
            v3 = l3_enc.encode(get_obs_for_domain(domain, pair), epistemic=None)
            if v2: train_l2.append(_pad(v2[0]))
            if v3: train_l3.append(_pad(v3[0]))
        except: continue
    if not train_l2 or not train_l3: return 0
    a2, a3 = np.mean(train_l2, axis=0), np.mean(train_l3, axis=0)
    test_in = task.get('test_input') or (task['test']['init'] if domain == 'phyre' else None)
    candidates = task.get('candidates') or (task['test']['candidates'] if domain == 'phyre' else [])
    scores = []
    for cand in candidates:
        try:
            cf = cand['final'] if domain == 'phyre' else cand
            v2 = l2_enc.encode((test_in, cf), epistemic=None)
            v3 = l3_enc.encode(get_obs_for_domain(domain, (test_in, cf)), epistemic=None)
            if not v2 or not v3: scores.append(-1e9); continue
            dl2, dla = _pad(v2[0]) - a2, _pad(v3[0]) - a3
            scores.append(-float(np.linalg.norm(dla - M_shared @ dl2)))
        except: scores.append(-1e9)
    return int(np.argmax(scores))

def run_transfer_experiment(source_domains: list[str], target_domain: str, n: int = 15) -> dict:
    rng = np.random.default_rng(42)
    matrices = []
    for d in source_domains:
        l2e, l3e = get_encoders(d)
        tasks = load_domain_tasks(d, n, rng)
        matrices.append(fit_domain_matrix(compute_delta_pairs(tasks, d, l2e, l3e)))
    M_global, _ = align_multiple_domains(matrices, ref_idx=0)
    test_tasks = load_domain_tasks(target_domain, n, rng)
    tl2e, tl3e = get_encoders(target_domain)
    l2l3_c, delta_c = 0, 0
    for t in test_tasks:
        correct = t['test']['correct_idx'] if target_domain == 'phyre' else t.get('correct_idx', 0)
        if score_l2l3_baseline(t, target_domain, tl2e, tl3e) == correct: l2l3_c += 1
        if score_with_anchor(t, target_domain, M_global, tl2e, tl3e) == correct: delta_c += 1
    count = len(test_tasks)
    return {'l2l3': l2l3_c / count if count else 0, 'delta': delta_c / count if count else 0}

# --------------------------------------------------------------------------- #
#  Phase 4: Surprise Transfer
# --------------------------------------------------------------------------- #

def measure_surprise(domain: str, n: int = 10) -> float:
    from hpm.agents.l4_generative import L4GenerativeHead
    from hpm.agents.l5_monitor import L5MetaMonitor
    l2e, l3e = get_encoders(domain)
    if domain == 'chem':
        from benchmarks.chem_logic_sim import generate_ambiguous_chem_tasks, generate_chem_tasks
        train_tasks = generate_chem_tasks(n_tasks=n, seed=42)
        test_tasks = [t for t in generate_ambiguous_chem_tasks(n_tasks=n, seed=43) if t['scenario'] == 'latent_ph']
    elif domain == 'linguistic':
        from benchmarks.linguistic_sim import generate_register_tasks
        tasks = generate_register_tasks(n_train=n, seed=42)
        train_tasks, test_tasks = [t for t in tasks if not t.get('is_trap')], [t for t in tasks if t.get('is_trap')]
    else: return 0.0
    head = L4GenerativeHead(l2e.feature_dim, l3e.feature_dim)
    for t in train_tasks:
        obs = (t['reactant'], t['product'])
        head.accumulate(_encode_vec(l2e, obs), _encode_vec(l3e, obs))
    head.fit()
    monitor = L5MetaMonitor()
    for t in test_tasks:
        l2_in, l3_act = _encode_vec(l2e, (t['reactant'], t['product'])), _encode_vec(l3e, (t['reactant'], t['product']))
        l4_pred = head.predict(l2_in)
        if l4_pred is not None: monitor.update(l4_pred, l3_act)
    return float(np.mean(monitor._surprises)) if monitor._surprises else 0.0

def run_surprise_transfer():
    print("\nSP15 Surprise Transfer (Metacognitive Alignment)")
    print(f"{'Source':<15} {'Target':<15} {'Surprise':>10} {'Portable?':>12}")
    print("-" * 60)
    s_ling, s_chem = measure_surprise('linguistic'), measure_surprise('chem')
    print(f"{'Linguistic':<15} {'Chemistry':<15} {s_ling:>10.3f} {'YES' if abs(s_ling-s_chem)<0.2 else 'PARTIAL':>12}")
    print(f"{'Chemistry':<15} {'Linguistic':<15} {s_chem:>10.3f} {'YES' if abs(s_ling-s_chem)<0.2 else 'PARTIAL':>12}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SP15 Generalized Cross-Domain Transfer")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()
    n_tasks = 4 if args.smoke else 15
    exps = [(['phyre', 'arc', 'ds1000'], 'chem'), (['ds1000', 'chem'], 'math'), (['math', 'phyre', 'arc', 'ds1000', 'chem'], 'linguistic')]
    print("\nSP15 Generalized Cross-Domain Transfer")
    print(f"{'Sources':<40} {'Target':<12} {'l2l3':>6} {'delta':>10}")
    print("-" * 72)
    for src, tgt in exps:
        res = run_transfer_experiment(src, tgt, n=n_tasks)
        label = " + ".join(s.capitalize() for s in src)
        if len(label) > 38: label = label[:35] + "..."
        print(f"{label:<40} {tgt:<12} {res['l2l3']:>6.3f} {res['delta']:>10.3f}")
    run_surprise_transfer()

if __name__ == "__main__":
    main()
