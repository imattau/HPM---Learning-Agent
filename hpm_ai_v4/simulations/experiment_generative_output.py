#!/usr/bin/env python3
"""Generative Output Experiment — uses LayeredAgent for actual character output."""
import argparse
import numpy as np
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.full_simulation import WikipediaStream

CLASS_NAMES = ['letter', 'digit', 'space', 'punct', 'newline']


def run_generative_demo(corpus_path: str, steps: int = 10000):
    print(f"\n[output] Generative demo: {corpus_path!r} ({steps} steps)")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    # Collect tokens for training and context
    tokens = []
    for _ in range(steps + 100):
        tokens.append(next(stream_iter))

    layered = LayeredAgent(num_workers=1)

    print(f"[learn] Training on {steps} tokens...")
    for i in range(steps):
        layered.perceive(tokens[i])
        if (i + 1) % 2000 == 0:
            print(f"  Step {i+1}...")

    print("\n" + "=" * 50)
    print("DEMONSTRATING OUTPUT CAPABILITIES")
    print("=" * 50)

    # 1. L2 Character Prediction
    context_raw = tokens[steps - 20:steps]
    context_str = "".join(chr(v + 32) for v in context_raw if 0 <= v <= 94)
    print(f"\n[1. Prediction] Context: '{context_str}'")
    preds = layered.predict_next_chars(context_raw, top_k=5)
    print("  Next character predictions (L2):")
    for ch, prob in preds:
        print(f"    {repr(ch)}: {prob*100:.1f}%")

    # 2. L2 Character Generation
    print(f"\n[2. Generation] L2 generates 80 characters:")
    generated = layered.generate(steps=80)
    print(f"  '{generated}'")
    words = [w for w in generated.split() if len(w) >= 2]
    print(f"  Words (len>=2): {words[:10]}")

    # 3. L1 Pattern Explanation
    print(f"\n[3. L1 Patterns] Dominant char-class patterns:")
    top3_l1 = sorted(layered.l1.patterns, key=lambda p: -p.weight)[:3]
    for i, p in enumerate(top3_l1):
        # Sample most likely observation from most likely initial state
        top_obs = int(np.argmax(p.B[np.argmax(p.pi)]))
        top_cls = CLASS_NAMES[min(top_obs, 4)]
        print(f"  L1 Pattern {i+1} (weight={p.weight:.3f} K={p.latent_dim}): "
              f"predicts '{top_cls}'  MI={p.compression():.3f}")

    # 4. L2 Pattern Explanation
    print(f"\n[4. L2 Patterns] Dominant char-level patterns:")
    top3_l2 = sorted(layered.l2.patterns, key=lambda p: -p.weight)[:3]
    for i, p in enumerate(top3_l2):
        top_obs = int(np.argmax(p.B[np.argmax(p.pi)]))
        top_ch = repr(chr(top_obs + 32)) if 0 <= top_obs <= 94 else '?'
        print(f"  L2 Pattern {i+1} (weight={p.weight:.3f} K={p.latent_dim}): "
              f"predicts {top_ch}  MI={p.compression():.3f}")

    # 5. Planning (find space after "the")
    prefix_raw = [ord(ch) - 32 for ch in "the"]
    goal = ord(' ') - 32  # space = 0
    print(f"\n[5. Planning] From 'the' find space (horizon=6, rollouts=30):")
    # Setup context for planning
    layered.l2.obs_buffer = list(prefix_raw)
    plan = layered.l2.reasoner.plan(goal_state=goal, horizon=6, num_rollouts=30,
                                     require_valid_words=False, require_grammatical=False)
    plan_str = "".join(chr(v + 32) for v in plan if 0 <= v <= 94)
    reached = (plan[-1] == goal) if plan else False
    print(f"  Plan: 'the{plan_str}'  reached_space={'YES' if reached else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Output Experiment")
    parser.add_argument('--corpus', default="data/history_of_science.txt")
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()
    run_generative_demo(args.corpus, steps=args.steps)
