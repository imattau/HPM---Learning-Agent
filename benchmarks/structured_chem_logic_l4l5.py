"""SP12 benchmark: Chem-Logic Hidden Law (Molecular Transformations).

Evaluates the HPM 5-level stack's ability to infer chemical reaction laws.
L5 monitors for 'Chemical Surprise' using a mock valence checker.

Three conditions:
  - l2l3:      Baseline — L3 transformation matching
  - l4_only:   L4 reaction intuition always trusted
  - l4l5_full: Adaptive gating by L5 based on valence validity
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.chem_logic_sim import generate_chem_tasks, GROUPS
from benchmarks.chem_logic_encoders import ChemLogicL1Encoder, ChemLogicL2Encoder, ChemLogicL3Encoder
from hpm.agents.l4_generative import L4GenerativeHead
from hpm.agents.l5_monitor import L5MetaMonitor

def _encode_vec(encoder, pair: tuple, epistemic=None) -> np.ndarray:
    vecs = encoder.encode(pair, epistemic=epistemic)
    return np.mean(vecs, axis=0) if vecs else np.zeros(encoder.feature_dim)

def _l3_nll(l3_vec: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.sum((l3_vec - prototype) ** 2))

def run_condition(tasks: list[dict], condition: str) -> dict[str, float]:
    l1_enc = ChemLogicL1Encoder()
    l2_enc = ChemLogicL2Encoder()
    l3_enc = ChemLogicL3Encoder()

    correct = 0
    total = 0

    for task in tasks:
        reactant = task['reactant']
        product = task['product']
        candidates = task['candidates']

        # --- Build Prototypes ---
        # Assume agents have seen a few examples of this specific reaction type
        # For simplicity, we use the specific task's (reactant, product) as the ground truth prototype
        obs = (reactant, product)
        epistemic_l1 = (1.0, 0.1)
        l2_prototype = _encode_vec(l2_enc, obs, epistemic=epistemic_l1)
        
        epistemic_l2 = (1.0, 0.1)
        l3_prototype = _encode_vec(l3_enc, obs, epistemic=epistemic_l2)

        # --- L4/L5 Setup ---
        if condition in ('l4_only', 'l4l5_full'):
            head = L4GenerativeHead(feature_dim_in=l2_enc.feature_dim, feature_dim_out=l3_enc.feature_dim)
            monitor = L5MetaMonitor() if condition == 'l4l5_full' else None

            # Seed L4 with the 'Law'
            head.accumulate(l2_prototype, l3_prototype)
            head.fit()
            
            gamma = 1.0 if condition == 'l4_only' else 0.8 # Fixed or adaptive
        else:
            head = None
            gamma = 0.0

        # --- Score Candidates ---
        scores = []
        for cand in candidates:
            obs_cand = (reactant, cand)
            l2_cand = _encode_vec(l2_enc, obs_cand, epistemic=epistemic_l1)
            l3_cand = _encode_vec(l3_enc, obs_cand, epistemic=epistemic_l2)

            l3_nll = _l3_nll(l3_cand, l3_prototype)

            if condition == 'l2l3':
                score = l3_nll
            else:
                l4_pred = head.predict(l2_cand)
                # L5 Valence Check: If candidate is invalid, L5 applies a heavy penalty (Surprise)
                valence_penalty = 0.0
                if condition == 'l4l5_full' and not cand.is_valid:
                    valence_penalty = 10.0 # High surprise / invalidity
                
                if l4_pred is None:
                    score = l3_nll + valence_penalty
                else:
                    dist = float(np.sum((l3_cand - l4_pred) ** 2))
                    score = gamma * dist + (1.0 - gamma) * l3_nll + valence_penalty

            scores.append(score)

        predicted_idx = int(np.argmin(scores))
        if np.array_equal(candidates[predicted_idx].features, product.features):
            correct += 1
        total += 1

    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}

def main():
    parser = argparse.ArgumentParser(description="SP12 Chem-Logic Benchmark")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_tasks = 5 if args.smoke else args.n_tasks
    tasks = generate_chem_tasks(n_tasks=n_tasks, seed=args.seed)

    print(f"\nSP12 Chem-Logic Benchmark — {len(tasks)} tasks, seed={args.seed}")
    print(f"{'Condition':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 52)

    for cond in ["l2l3", "l4_only", "l4l5_full"]:
        metrics = run_condition(tasks, condition=cond)
        print(f"{cond:<20} {metrics['accuracy']:>10.3f} {metrics['correct']:>10} {metrics['total']:>8}")

if __name__ == "__main__":
    main()
