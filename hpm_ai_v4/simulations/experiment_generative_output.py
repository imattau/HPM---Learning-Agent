#!/usr/bin/env python3
"""
Generative Output Experiment.
Demonstrates HPM v4's ability to output information via prediction, simulation, and explanation.
Now supports dictionary-guided planning and evaluation.
"""
import argparse
import numpy as np
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import NLTKGrammarLibrary

CLASS_NAMES = ['letter', 'digit', 'space', 'punct', 'newline']
CLASS_SYMBOLS = ['L', 'D', '_', '.', '/']  # compact for simulation display


def run_generative_demo(corpus_path: str, steps: int = 5000, use_dictionary: bool = False, use_grammar: bool = False):
    print(f"\n[output] Starting generative demo on {corpus_path!r} ({steps} steps, obs_dim=5 char classes)")

    dictionary = NLTKWordList() if use_dictionary else None
    grammar = NLTKGrammarLibrary() if use_grammar else None

    adapter = CharClassAdapter()
    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    tokens = [adapter.encode(next(stream_iter)) for _ in range(steps + 100)]

    # Equal weights: hier patterns compete fairly against flat baseline
    agent = HPMAgent(num_initial_patterns=4, obs_dim=5, dictionary=dictionary, grammar=grammar)
    agent.patterns = []
    for i in range(4):
        p = HierarchicalPattern(i, latent_dim=2, obs_dim=5)
        p.weight = 0.15
        agent.patterns.append(p)
    for i in range(4, 6):
        p = FlatPattern(i, obs_dim=5)
        p.weight = 0.1
        agent.patterns.append(p)
    
    print(f"[learn] Training on {steps} character tokens...")
    for i in range(steps):
        agent.perceive_and_learn(tokens[i])
        if (i + 1) % 2000 == 0:
            print(f"  Step {i+1}...")

    print("\n" + "=" * 50)
    print("DEMONSTRATING OUTPUT CAPABILITIES")
    print("=" * 50)

    # A. PREDICTION (Next Class)
    last_context = tokens[steps - 20:steps]
    context_str = "".join(CLASS_SYMBOLS[t] for t in last_context)
    print(f"\n[1. Prediction] Context (L=letter D=digit _=space .=punct /=newline):")
    print(f"  '{context_str}'")
    relevant = agent.reasoner.get_relevant_patterns(last_context)
    dist = agent.reasoner.compose_predictions(relevant, last_context)
    top_indices = np.argsort(dist)[::-1]
    print("  Next class predictions:")
    for idx in top_indices:
        print(f"    {CLASS_NAMES[idx]:<10}: {dist[idx]*100:.1f}%")

    # B. SIMULATION (Imagined Future — class sequence)
    print(f"\n[2. Simulation] Imagining next 40 class tokens...")
    future_tokens = agent.reasoner.simulate_future(steps=40)
    future_syms = "".join(CLASS_SYMBOLS[min(t, 4)] for t in future_tokens)
    # Count word-like runs (letter sequences between spaces)
    words = future_syms.replace('_', ' ').split()
    print(f"  Class sequence: '{future_syms}'")
    print(f"  Word-like runs: {len(words)} ({', '.join(w for w in words[:5])}...)")

    # C. EXPLANATION
    print(f"\n[3. Explanation] Dominant patterns:")
    for i, p in enumerate(relevant[:3]):
        top_obs = int(np.argmax(p.B[np.argmax(p.pi)]))
        top_cls = CLASS_NAMES[min(top_obs, 4)]
        print(f"  Pattern {i+1} (weight={p.weight:.3f} K={p.latent_dim}): "
              f"predicts '{top_cls}'  MI={p.compression():.3f}")

    # D. PLANNING (find 'space' class from letter context)
    prefix_classes = [0, 0, 0, 2]  # L L L _ (e.g. "the ")
    goal_class = 2  # space
    print(f"\n[4. Planning] From [L,L,L,_] find next space (horizon=6, rollouts=30)...")
    agent.obs_buffer = list(prefix_classes)
    plan = agent.reasoner.plan(goal_state=goal_class, horizon=6, num_rollouts=30,
                               require_valid_words=False, require_grammatical=False)
    plan_syms = "".join(CLASS_SYMBOLS[min(t, 4)] for t in plan)
    reached = plan[-1] == goal_class if plan else False
    print(f"  Plan: 'LLL_{plan_syms}'  reached_space={'YES' if reached else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Output Experiment")
    parser.add_argument('--corpus', default="hpm_ai_v4/simulations/data/wiki_sample.txt")
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--use-dictionary', action='store_true')
    parser.add_argument('--use-grammar', action='store_true')
    args = parser.parse_args()

    run_generative_demo(args.corpus, steps=args.steps,
                        use_dictionary=args.use_dictionary,
                        use_grammar=args.use_grammar)
