"""
verify_l5_lm_synergy.py - The "Synergy Push" Experiment.
Compares Baseline vs LM-Only vs Full (LM+L5) agents.
Demonstrates that L5 Meta-Cognition acts as a force multiplier for LM-driven discovery.
"""

import os
import time
import numpy as np
import torch
from typing import Dict, Any, List

from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.curriculum import CurriculumManager
from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
from hpm_ai_v3.meta_cognitive_pattern import MetaCognitivePattern
from hpm_ai_v3.agents.meta_training import MetaTrainingLoop
from hpm_ai_v3.diagnostics import HPMLogger


def run_training_v3(name: str, use_lm: bool, use_l5: bool, max_episodes: int = 1000):
    print(f"\n>>> Starting Training: {name} (LM={use_lm}, L5={use_l5})")
    
    log_dir = f"./logs/synergy_{name.lower().replace(' ', '_')}"
    logger = HPMLogger(log_dir=log_dir)
    
    # 1. Initialize LM if used
    lm = None
    if use_lm:
        lm = LanguageModelPattern()
        corpus_path = "hpm_ai_v3/data/lm_corpus/rich_corpus.txt"
        if os.path.exists(corpus_path):
            print(f"  [LM] Pretraining on rich corpus...")
            lm.pretrain(corpus_path, epochs=10)
            
    # 2. Initialize Agent
    agent = UnifiedDiscoveryAgent(context_dim=64, lm=lm)
    curriculum = CurriculumManager()
    
    # 3. Initialize L5 if used
    meta = None
    meta_loop = None
    if use_l5:
        meta = MetaCognitivePattern()
        # Bind tool selector for cache consistency if LM is active
        if lm and hasattr(agent, 'tool_selector'):
            meta._tool_selector = agent.tool_selector
            lm._tool_selector = agent.tool_selector
            
        # Wrap in MetaTrainingLoop
        meta_loop = MetaTrainingLoop(agent, curriculum, meta, N=10)

    start_time = time.time()
    
    ep = 0
    while ep < max_episodes:
        if use_l5:
            # We want to log each episode within the meta-step for consistent stats
            meta_loop.run_meta_step()
            
            # Pull the 10 most recent rewards from the agent's meta-history
            recent_rewards = agent._meta_success_history[-10:] if agent._meta_success_history else [0.0]*10
            
            phase_name = curriculum.patterns[curriculum.active_pattern_idx].name
            for i, r in enumerate(recent_rewards):
                logger.log_episode(ep + i, phase_name, "meta_guided", r, agent.population)
            ep += 10
        else:
            task = curriculum.get_current_task()
            solution = agent.run_episode(task, max_steps=10)
            reward = agent.evaluate_solution(solution)
            
            # Record diagnostics
            phase_name = curriculum.patterns[curriculum.active_pattern_idx].name
            logger.log_episode(ep, phase_name, "standard", reward, agent.population)
            
            # Update curriculum
            curriculum.update(reward)
            ep += 1

        if ep % 100 == 0:
            phase_name = curriculum.patterns[curriculum.active_pattern_idx].name
            success = 1 if (agent._meta_success_history[-1] if use_l5 else reward) > 0.8 else 0
            print(f"  [Ep {ep}] Phase: {phase_name} | Success: {success}")

    total_time = time.time() - start_time
    summary = logger.get_summary()
    print(f"\n>>> {name} Complete in {total_time:.1f}s")
    print(f"    Summary: {summary}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    
    results = {}
    
    # 1. Baseline (Control)
    if not args.skip_baseline:
        results["Baseline"] = run_training_v3("Baseline", use_lm=False, use_l5=False, max_episodes=args.episodes)
    
    # 2. LM-Only (Neural Intuition)
    results["LM-Only"] = run_training_v3("LM-Only", use_lm=True, use_l5=False, max_episodes=args.episodes)
    
    # 3. LM + L5 (Neural Intuition + Strategic Oversight)
    results["Full (LM+L5)"] = run_training_v3("Full (LM+L5)", use_lm=True, use_l5=True, max_episodes=args.episodes)
    
    print("\n" + "="*60)
    print(f"{'AGENT':<15} | {'MAX PHASE':<25} | {'SUCCESS RATE':<15}")
    print("-"*60)
    for name, summary in results.items():
        phase = summary.get('last_phase', 'N/A')
        success = summary.get('success_rate_recent', 0.0)
        print(f"{name:<15} | {phase:<25} | {success:>13.2%}")
    print("="*60)
    
    # Evaluation Logic
    full = results["Full (LM+L5)"]
    lm_only = results["LM-Only"]
    
    print("\nSYNERGY EVALUATION:")
    if full.get('success_rate_recent', 0) > lm_only.get('success_rate_recent', 0) + 0.02:
        print(">>> SUCCESS: L5 Meta-Cognition provided a significant performance boost.")
    elif full.get('last_phase') != lm_only.get('last_phase'):
        print(">>> SUCCESS: L5 Meta-Cognition enabled reaching a more advanced phase.")
    else:
        print(">>> INCONCLUSIVE: L5 did not show a clear advantage in this run.")
