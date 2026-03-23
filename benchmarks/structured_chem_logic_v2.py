"""SP13 benchmark: Chem-Logic II (Ambiguity & Competition).

Tests HPM stack on non-deterministic chemical reasoning:
1. Competitive Inhibition: Multiple reactive sites with priority rules.
2. Latent pH: Unobserved environmental variables causing outcome divergence.

Evaluates how L5 'Surprise' detects latent shifts and priority conflicts.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.chem_logic_sim import generate_ambiguous_chem_tasks, GROUPS, get_molecule
from benchmarks.chem_logic_encoders import ChemLogicL1Encoder, ChemLogicL2Encoder, ChemLogicL3Encoder
from hpm.agents.l4_generative import L4GenerativeHead
from hpm.agents.l5_monitor import L5MetaMonitor

def _encode_vec(encoder, pair: tuple, epistemic=None) -> np.ndarray:
    vecs = encoder.encode(pair, epistemic=epistemic)
    return np.mean(vecs, axis=0) if vecs else np.zeros(encoder.feature_dim)

def _l3_nll(l3_vec: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.sum((l3_vec - prototype) ** 2))

def run_v2_condition(tasks: list[dict], condition: str) -> dict:
    l1_enc = ChemLogicL1Encoder()
    l2_enc = ChemLogicL2Encoder()
    l3_enc = ChemLogicL3Encoder()

    correct = 0
    total = 0
    total_surprise = 0.0
    surprise_count = 0

    scenario_metrics = {"competition": {"correct": 0, "total": 0}, "latent_ph": {"correct": 0, "total": 0}}

    # Pre-train L4 on standard rules (High pH / Single Site)
    head = L4GenerativeHead(feature_dim_in=l2_enc.feature_dim, feature_dim_out=l3_enc.feature_dim)
    monitor = L5MetaMonitor() if "l4l5" in condition else None

    # Deterministic Training Set (learned rules)
    # 1. Normal Amine reaction (Methylation)
    r1, p1 = get_molecule("N"), get_molecule("CN") 
    l3_amine = _encode_vec(l3_enc, (r1, p1))
    head.accumulate(_encode_vec(l2_enc, (r1, p1)), l3_amine)
    # 2. Normal Hydroxyl reaction (Methylation)
    r2, p2 = get_molecule("O"), get_molecule("CO")
    l3_hydroxyl = _encode_vec(l3_enc, (r2, p2))
    head.accumulate(_encode_vec(l2_enc, (r2, p2)), l3_hydroxyl)
    head.fit()

    # The agent's 'Rule Store' (L3 patterns) - only knows the normal reactions
    known_rules = [l3_amine, l3_hydroxyl]

    for task in tasks:
        reactant = task['reactant']
        product = task['product']
        candidates = task['candidates']
        scenario = task['scenario']

        epistemic_l1 = (1.0, 0.1)
        epistemic_l2 = (1.0, 0.1)
        
        # Test input L2
        l2_in = _encode_vec(l2_enc, (reactant, product), epistemic=epistemic_l1)

        # --- Score Candidates ---
        scores = []
        for cand in candidates:
            obs_cand = (reactant, cand)
            l2_cand = _encode_vec(l2_enc, obs_cand, epistemic=epistemic_l1)
            l3_cand = _encode_vec(l3_enc, obs_cand, epistemic=epistemic_l2)

            # L3 Matching: Match against the BEST known rule (Analytical)
            # This simulates an agent that has a library of rules and tries to find one that fits.
            l3_nll = min(_l3_nll(l3_cand, rule) for rule in known_rules)

            if condition == 'l2l3':
                score = l3_nll
            elif condition == 'l4_only':
                l4_pred = head.predict(l2_in)
                if l4_pred is None: score = l3_nll
                else: score = float(np.sum((l3_cand - l4_pred) ** 2)) 
            else: # l4l5_full
                l4_pred = head.predict(l2_in)
                valence_penalty = 0.0
                if not cand.is_valid:
                    valence_penalty = 15.0 
                
                if l4_pred is None:
                    score = l3_nll + valence_penalty
                else:
                    dist = float(np.sum((l3_cand - l4_pred) ** 2))
                    
                    # Surprise Check: If intuition fails to predict the actual outcome, monitor surprise
                    if np.array_equal(cand.features, product.features):
                        monitor.update(l4_pred, l3_cand)
                        if monitor._surprises:
                            surprise = monitor._surprises[-1]
                            total_surprise += surprise
                            surprise_count += 1
                    
                    gamma = monitor.strategic_confidence()
                    score = gamma * dist + (1.0 - gamma) * l3_nll + valence_penalty

            scores.append(score)

        predicted_idx = int(np.argmin(scores))
        cand_pred = candidates[predicted_idx]
        is_correct = np.array_equal(cand_pred.features, product.features)
        
        if is_correct:
            correct += 1
            scenario_metrics[scenario]["correct"] += 1
        
        total += 1
        scenario_metrics[scenario]["total"] += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "avg_surprise": total_surprise / surprise_count if surprise_count > 0 else 0.0,
        "scenarios": scenario_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="SP13 Chem-Logic II Benchmark")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_tasks = 6 if args.smoke else args.n_tasks
    tasks = generate_ambiguous_chem_tasks(n_tasks=n_tasks, seed=args.seed)

    print(f"\nSP13 Chem-Logic II — {len(tasks)} ambiguous tasks, seed={args.seed}")
    print(f"{'Condition':<15} {'Accuracy':>10} {'Surprise':>10} {'Comp Acc':>10} {'pH Acc':>10}")
    print("-" * 65)

    for cond in ["l2l3", "l4_only", "l4l5_full"]:
        m = run_v2_condition(tasks, condition=cond)
        comp_acc = m["scenarios"]["competition"]["correct"] / m["scenarios"]["competition"]["total"] if m["scenarios"]["competition"]["total"] > 0 else 0
        ph_acc = m["scenarios"]["latent_ph"]["correct"] / m["scenarios"]["latent_ph"]["total"] if m["scenarios"]["latent_ph"]["total"] > 0 else 0
        
        print(f"{cond:<15} {m['accuracy']:>10.3f} {m['avg_surprise']:>10.3f} {comp_acc:>10.3f} {ph_acc:>10.3f}")

if __name__ == "__main__":
    main()
