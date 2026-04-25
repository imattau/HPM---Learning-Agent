"""
run_two_phase_experiment.py - Integrates task generation and two-phase training.
"""

import os
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.tools.python_substrate import register_python_substrate
from hpm_ai_v3.tools.memory import register_memory_tools
from hpm_ai_v3.tools.innate import register_innate_tools
from task_generator import generate_phase1_dataset
from two_phase_training import TwoPhaseTrainer

def main():
    print("=== HPM Two-Phase Training Experiment ===\n")
    
    # 1. Setup Agent and Environment
    register_python_substrate()
    register_innate_tools()
    
    checkpoint_dir = "./checkpoints/two_phase"
    os.makedirs(checkpoint_dir, exist_ok=True)
    register_memory_tools(persistence_path=os.path.join(checkpoint_dir, "kv_store.json"))
    
    agent = UnifiedDiscoveryAgent(context_dim=64)
    trainer = TwoPhaseTrainer(agent)
    
    # 2. Generate Data
    print("Generating training and test datasets...")
    # 50 tasks for deep discovery, 10 for generalization test
    phase1_tasks = generate_phase1_dataset(50) 
    phase2_tasks = generate_phase1_dataset(10)
    
    # 3. Phase 1: Discovery with Answers (Persistence)
    print("\nStarting Phase 1...")
    trainer.run_phase1(phase1_tasks, max_steps_per_task=30)
    
    # 4. Phase 2: Application without Answers (Unsupervised)
    print("\nStarting Phase 2...")
    results = trainer.run_phase2(phase2_tasks, max_steps_per_task=30, use_memory=True)
    
    # 5. Summary
    print("\n=== Final Results Summary ===")
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    print(f"Phase 2 Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    
    for i, r in enumerate(results):
        status = "✓" if r['correct'] else "✗"
        print(f"[{i+1}] {r['task']} -> {r['solution']} (Expected: {r['expected']}) {status}")

if __name__ == "__main__":
    main()
