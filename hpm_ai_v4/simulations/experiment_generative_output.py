#!/usr/bin/env python3
"""
Generative Output Experiment.
Demonstrates HPM v4's ability to output information via prediction, simulation, and explanation.
Now supports dictionary-guided planning and evaluation.
"""
import time
import argparse
import numpy as np
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import TextAdapter
from hpm_ai_v4.simulations.build_library import fetch_wikipedia_content
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import NLTKGrammarLibrary


def run_generative_demo(topic: str, steps: int = 2000, use_dictionary: bool = False, use_grammar: bool = False):
    print(f"\n[output] Starting {('Dictionary ' if use_dictionary else '') + ('Grammar ' if use_grammar else '')}Guided Generative Demo for topic: {topic!r}")
    
    # 0. Initialize Libraries (optional)
    dictionary = NLTKWordList() if use_dictionary else None
    grammar = NLTKGrammarLibrary() if use_grammar else None
    
    # 1. Fetch and Prepare
    corpus = fetch_wikipedia_content(topic)
    if not corpus:
        return
    
    adapter = TextAdapter()
    tokens = adapter.to_observations(corpus, max_length=steps + 100)
    
    # 2. Initialize Agent
    # Using larger obs_dim (256) for character-level text
    agent = HPMAgent(num_initial_patterns=3, obs_dim=256, dictionary=dictionary, grammar=grammar)
    
    print(f"[learn] Training on {steps} character tokens...")
    for i in range(steps):
        agent.perceive_and_learn(tokens[i])
        if (i + 1) % 500 == 0:
            print(f"  Step {i+1}...")

    print("\n" + "="*50)
    print("DEMONSTRATING OUTPUT CAPABILITIES")
    print("="*50)

    # A. PREDICTION (Next Character)
    last_context = tokens[steps:steps+20]
    print(f"\n[1. Prediction] Context: '{corpus[steps:steps+20]}'")
    
    # Get blended prediction
    relevant = agent.reasoner.get_relevant_patterns(last_context)
    dist = agent.reasoner.compose_predictions(relevant, last_context)
    
    # Top 3 predicted characters
    top_indices = np.argsort(dist)[-3:][::-1]
    print("  Next character predictions:")
    for idx in top_indices:
        char = adapter.reverse_vocab.get(idx, f"<{idx}>")
        prob = dist[idx]
        print(f"    '{char}': {prob*100:.1f}%")

    # B. SIMULATION (Imagined Future / Hallucination)
    print(f"\n[2. Simulation] Imagining next 40 characters...")
    future_tokens = agent.reasoner.simulate_future(steps=40)
    future_text = "".join([adapter.reverse_vocab.get(t, "?") for t in future_tokens])
    print(f"  Generated sequence: \"{future_text}\"")

    # C. EXPLANATION (Pattern Interpretation)
    print(f"\n[3. Explanation] Interpreting dominant patterns:")
    for i, p in enumerate(relevant[:2]):
        explanation = agent.reasoner.explain(p)
        print(f"  Pattern {i+1} (Weight {p.weight:.3f}): {explanation}")

    # D. PLANNING (Word Completion)
    # Context: "Artificia" -> Goal: 'l'
    prefix = "Artificia"
    prefix_tokens = adapter.to_observations(prefix)
    # Goal is 'l' (ord('l')-32 = 76)
    goal_char = 'l'
    goal_token = adapter.vocab.get(goal_char)
    
    print(f"\n[4. Planning] Completing the word '{prefix}' to find '{goal_char}'...")
    agent.obs_buffer = list(prefix_tokens) # Fake buffer for planning
    plan = agent.reasoner.plan(goal_state=goal_token, horizon=5, num_rollouts=20, 
                               require_valid_words=use_dictionary,
                               require_grammatical=use_grammar)
    
    plan_chars = "".join([adapter.reverse_vocab.get(t, "?") for t in plan])
    print(f"  Plan to reach '{goal_char}': \"{plan_chars}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Output Experiment")
    parser.add_argument('--topic', default="Artificial Intelligence", help='Wikipedia topic')
    parser.add_argument('--steps', type=int, default=3000, help='Total training steps')
    parser.add_argument('--use-dictionary', action='store_true', help='Enable NLTK dictionary guidance')
    parser.add_argument('--use-grammar', action='store_true', help='Enable Heuristic Grammar guidance')
    args = parser.parse_args()

    run_generative_demo(args.topic, steps=args.steps, use_dictionary=args.use_dictionary, use_grammar=args.use_grammar)
