"""SP14 benchmark: Linguistic Register Shift (The Social pH).

Tests HPM stack on detecting hidden shifts in social register (Formal vs Informal).
L5 monitor tracks surprise when Formal intuition fails on Informal test data.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.linguistic_sim import generate_register_tasks
from benchmarks.linguistic_encoders import LinguisticL1Encoder, LinguisticL2Encoder, LinguisticL3Encoder
from hpm.agents.l4_generative import L4GenerativeHead
from hpm.agents.l5_monitor import L5MetaMonitor

def _encode_vec(encoder, pair: tuple, epistemic=None) -> np.ndarray:
    vecs = encoder.encode(pair, epistemic=epistemic)
    return np.mean(vecs, axis=0) if vecs else np.zeros(encoder.feature_dim)

def _l3_nll(l3_vec: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.sum((l3_vec - prototype) ** 2))

def run_register_condition(tasks: list[dict], condition: str) -> dict:
    l1_enc = LinguisticL1Encoder()
    l2_enc = LinguisticL2Encoder()
    l3_enc = LinguisticL3Encoder()

    correct = 0
    total = 0
    surprise_trap = 0.0

    # Separate train and test
    train_tasks = [t for t in tasks if not t.get("is_trap")]
    test_tasks = [t for t in tasks if t.get("is_trap")]

    # --- Global Rule Library (The agent knows these transformations exist) ---
    from benchmarks.linguistic_sim import get_word, REGISTER_MAP
    # Formal Rule
    v = "ask"
    r_f, p_f = get_word(v, is_root=True), get_word(REGISTER_MAP["formal"][v], register="formal")
    l3_formal = _encode_vec(l3_enc, (r_f, p_f))
    # Informal Rule
    r_i, p_i = get_word(v, is_root=True), get_word(REGISTER_MAP["informal"][v], register="informal")
    l3_informal = _encode_vec(l3_enc, (r_i, p_i))
    
    known_rules = [l3_formal, l3_informal]

    # --- Training Phase (Formal Register) ---
    # The agent's INTUITION (L4) is trained ONLY on Formal examples
    head = L4GenerativeHead(feature_dim_in=l2_enc.feature_dim, feature_dim_out=l3_enc.feature_dim)
    monitor = L5MetaMonitor() if "l4l5" in condition else None

    for t in train_tasks:
        obs = (t['reactant'], t['product'])
        l2_vec = _encode_vec(l2_enc, obs)
        l3_vec = _encode_vec(l3_enc, obs)
        head.accumulate(l2_vec, l3_vec)
        
    head.fit()

    # --- Test Phase (The Trap) ---
    for t in test_tasks:
        reactant = t['reactant']
        product = t['product']
        candidates = t['candidates']

        epistemic_l1 = (1.0, 0.1)
        epistemic_l2 = (1.0, 0.1)
        l2_in = _encode_vec(l2_enc, (reactant, product), epistemic=epistemic_l1)

        scores = []
        for cand in candidates:
            obs_cand = (reactant, cand)
            l2_cand = _encode_vec(l2_enc, obs_cand, epistemic=epistemic_l1)
            l3_cand = _encode_vec(l3_enc, obs_cand, epistemic=epistemic_l2)

            # L3 Matching: Match against the library of rules
            l3_nll = min(_l3_nll(l3_cand, rule) for rule in known_rules)

            if condition == 'l2l3':
                # Analytical baseline doesn't know which rule to prefer
                score = l3_nll
            elif condition == 'l4_only':
                l4_pred = head.predict(l2_in)
                if l4_pred is None: score = l3_nll
                else: score = float(np.sum((l3_cand - l4_pred) ** 2))
            else: # l4l5_full
                l4_pred = head.predict(l2_in)
                if l4_pred is None:
                    score = l3_nll
                else:
                    dist = float(np.sum((l3_cand - l4_pred) ** 2))
                    
                    # Surprise Check on the actual outcome (to simulate learning/monitoring)
                    if cand.text == product.text:
                        monitor.update(l4_pred, l3_cand)
                        if monitor._surprises:
                            surprise_trap = monitor._surprises[-1]
                    
                    gamma = monitor.strategic_confidence()
                    # High surprise -> Low gamma -> trust L3 analytical library
                    score = gamma * dist + (1.0 - gamma) * l3_nll

            scores.append(score)

        predicted_idx = int(np.argmin(scores))
        if candidates[predicted_idx].text == product.text:
            correct += 1
        total += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "surprise": surprise_trap
    }

def main():
    parser = argparse.ArgumentParser(description="SP14 Register Shift Benchmark")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_train", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Run multiple times to average surprise
    total_m = {"l2l3": {"acc": 0, "surp": 0}, "l4_only": {"acc": 0, "surp": 0}, "l4l5_full": {"acc": 0, "surp": 0}}
    n_runs = 5 if args.smoke else 20
    
    for r in range(n_runs):
        tasks = generate_register_tasks(n_train=args.n_train, seed=args.seed + r)
        for cond in total_m.keys():
            res = run_register_condition(tasks, cond)
            total_m[cond]["acc"] += res["accuracy"]
            total_m[cond]["surp"] += res["surprise"]

    print(f"\nSP14 Linguistic Register Shift — {n_runs} runs, n_train={args.n_train}")
    print(f"{'Condition':<15} {'Accuracy':>10} {'Surprise':>10}")
    print("-" * 40)

    for cond, m in total_m.items():
        print(f"{cond:<15} {m['acc']/n_runs:>10.3f} {m['surp']/n_runs:>10.3f}")

if __name__ == "__main__":
    main()
