#!/usr/bin/env python3
"""Generative Output Experiment — inspects the stacked hierarchy."""
import argparse
import numpy as np
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary

CLASS_NAMES = ['letter', 'digit', 'space', 'punct', 'newline']


def run_generative_demo(corpus_path: str, steps: int = 10000):
    print(f"\n[output] Generative demo: {corpus_path!r} ({steps} steps)")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    # Collect tokens for training and context
    tokens = []
    for _ in range(steps + 100):
        tokens.append(next(stream_iter))

    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    layered = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)

    print(f"[learn] Training on {steps} tokens...")
    for i in range(steps):
        layered.perceive(tokens[i])
        if (i + 1) % 2000 == 0:
            print(f"  Step {i+1}...")

    print("\n" + "=" * 50)
    print("DEMONSTRATING OUTPUT CAPABILITIES")
    print("=" * 50)

    # 1. L1 Character-Class Prediction
    context_raw = tokens[steps - 20:steps]
    context_str = "".join(chr(v + 32) for v in context_raw if 0 <= v <= 94)
    print(f"\n[1. Prediction] Context: '{context_str}'")
    preds = layered.predict_next_chars(context_raw, top_k=5)
    print("  Next character-class predictions (L1):")
    for ch, prob in preds:
        print(f"    {repr(ch)}: {prob*100:.1f}%")

    # 2. Readable Text Generation
    print(f"\n[2. Generation] Layered decoder generates readable text:")
    target_text = "".join(chr(v + 32) for v in tokens[steps:steps + 80] if 0 <= v <= 94)
    planned = layered.plan_text_continuation(
        target_text=target_text,
        seed_text=context_str,
        horizon=80,
        strategy="beam",
        lookback=240,
    )
    generated = planned or layered.generate_text(
        steps=80,
        seed_text=context_str,
        target_text=target_text,
        mode="target",
        include_seed=False,
    )
    print(f"  Planned continuation: '{planned}'")
    print(f"  '{generated}'")
    eval_stats = layered.evaluate_generated_text(generated, target_text)
    print(f"  Target agreement={eval_stats['token_agreement']:.3f} "
          f"plausibility={eval_stats['plausibility']:.3f}")

    hybrid_stats = layered.observe_text(
        target_text,
        feedback_mode="hybrid",
        generated_text=generated,
        self_feedback_weight=0.02,
    )
    print(f"  Hybrid feedback: target_chars={hybrid_stats['target_chars']} "
          f"self_chars={hybrid_stats['self_chars']}")

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
    print(f"\n[4. L2 Patterns] Dominant state-level patterns:")
    top3_l2 = sorted(layered.l2.patterns, key=lambda p: -p.weight)[:3]
    for i, p in enumerate(top3_l2):
        top_obs = int(np.argmax(p.B[np.argmax(p.pi)]))
        top_ch = f"state {top_obs}"
        print(f"  L2 Pattern {i+1} (weight={p.weight:.3f} K={p.latent_dim}): "
              f"predicts {top_ch}  MI={p.compression():.3f}")

    # 5. Planning on L2 latent state space
    prefix_raw = [ord(ch) - 32 for ch in "the"]
    for tok in prefix_raw:
        layered.perceive(tok)
    goal = 0
    print(f"\n[5. Planning] In L2 latent space, plan toward state 0:")
    plan = layered.l2.reasoner.plan(goal_state=goal, horizon=6, num_rollouts=30,
                                     require_valid_words=False, require_grammatical=False)
    plan_str = " ".join(str(v) for v in plan)
    reached = (plan[-1] == goal) if plan else False
    print(f"  Plan: {plan_str}  reached_state_0={'YES' if reached else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Output Experiment")
    parser.add_argument('--corpus', default="data/history_of_science.txt")
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()
    run_generative_demo(args.corpus, steps=args.steps)
