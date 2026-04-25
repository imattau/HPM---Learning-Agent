
import os
import time
from hpm_ai_v4.simulations.wikipedia_sim import run_simulation, VOCAB_SIZE
from hpm_ai_v4.simulations.text_reasoning import TextReasoningInterface
from hpm_ai_v4.simulations.data.get_corpus import DEFAULT_OUTPUT_PATH

def analyze_agent(tri: TextReasoningInterface):
    print("\n" + "="*50)
    print("COGNITIVE ANALYSIS")
    print("="*50)
    
    # 1. Self-Explanation
    print(f"\n[L3 Reasoning: Explain]")
    print(f"Best Pattern Description: {tri.explain_best_pattern()}")
    
    # 2. Next Char Prediction
    prefixes = ["the ", "quil", "to be ", "shall "]
    print(f"\n[L4 Reasoning: Prediction]")
    for p in prefixes:
        preds = tri.next_char_predict(p)
        pred_str = ", ".join([f"'{c}' ({prob:.2f})" for c, prob in preds])
        print(f"  '{p}' -> {pred_str}")
        
    # 3. Word Completion
    print(f"\n[L4 Reasoning: Word Completion]")
    for p in ["the", "qui", "sha", "wor"]:
        completed = tri.word_complete(p, max_chars=10)
        print(f"  '{p}' -> '{completed}'")
        
    # 4. Planning
    print(f"\n[L4 Reasoning: Planning to Boundary]")
    plan = tri.plan_to_boundary(horizon=10, num_rollouts=10)
    print(f"  Plan: '{plan}'")
    
    # 5. Counterfactuals
    print(f"\n[L4 Reasoning: Counterfactual Intervention]")
    context = "the "
    print(f"  Context: '{context}'")
    for forced in ["q", "k"]:
        shifted = tri.counterfactual_shift(context, forced)
        shift_str = ", ".join([f"'{c}' ({prob:.2f})" for c, prob in shifted])
        print(f"  If forced '{forced}', next-next is: {shift_str}")

if __name__ == "__main__":
    if not os.path.exists(DEFAULT_OUTPUT_PATH):
        print("Corpus not found. Please run get_corpus first.")
        exit(1)
        
    print("Starting Wikipedia/Shakespeare Simulation (Short Run for Analysis)...")
    start_time = time.time()
    
    # Run for 200 characters for a very quick but complete analysis
    agent, metrics = run_simulation(
        filepath=DEFAULT_OUTPUT_PATH,
        total_chars=200,
        num_initial_patterns=1,
        log_every=50
    )
    
    duration = time.time() - start_time
    print(f"\nSimulation completed in {duration:.2f}s")
    
    tri = TextReasoningInterface(agent)
    analyze_agent(tri)
