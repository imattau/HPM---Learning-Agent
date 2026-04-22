"""
verify_lm_integration.py - Side-by-side comparison of Baseline vs LM-Enabled agents.
Demonstrates that ToolSelector + LanguageModelPattern improves learning efficiency.
"""

import os
import time
import numpy as np
from typing import Dict, Any, List

from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.curriculum import CurriculumManager
from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
from hpm_ai_v3.diagnostics import HPMLogger


def run_training(name: str, use_lm: bool, max_episodes: int = 500):
    print(f"\n>>> Starting Training: {name} (use_lm={use_lm})")
    
    log_dir = f"./logs/verify_{name.lower().replace(' ', '_')}"
    logger = HPMLogger(log_dir=log_dir)
    
    # 1. Initialize LM if used
    lm = None
    if use_lm:
        lm = LanguageModelPattern()
        # Pretrain on rich corpus
        corpus_path = "hpm_ai_v3/data/lm_corpus/rich_corpus.txt"
        if os.path.exists(corpus_path):
            print(f"  [LM] Pretraining on rich corpus...")
            lm.pretrain(corpus_path, epochs=10)
        else:
            print("  [Warning] Rich corpus not found, skipping pretrain.")
            
    # 2. Initialize Agent
    agent = UnifiedDiscoveryAgent(context_dim=64, lm=lm)
    curriculum = CurriculumManager()
    
    start_time = time.time()
    
    for ep in range(max_episodes):
        task = curriculum.get_current_task()
        
        # Capture step-0 tool selection via a temporary hook
        step0_tool = "none"
        orig_act = agent.act
        def hook_act(*args, **kwargs):
            res = orig_act(*args, **kwargs)
            nonlocal step0_tool
            if step0_tool == "none":
                step0_tool = res.get("action", "unknown")
            return res
        agent.act = hook_act
        
        # Run episode
        solution = agent.run_episode(task, max_steps=10)
        reward = agent.evaluate_solution(solution)
        
        # Restore act
        agent.act = orig_act
        
        # Record diagnostics
        logger.log_episode(ep, curriculum.patterns[curriculum.active_pattern_idx].name, 
                           step0_tool, reward, agent.population)
        
        if use_lm and ep % 50 == 0:
            lm.validate_internal()
            logger.log_lm_stats(ep, lm.last_loss, lm.accuracy)
            
        # Update curriculum
        curriculum.update(reward)
        
        if ep % 100 == 0:
            print(f"  [Ep {ep}] Phase: {curriculum.patterns[curriculum.active_pattern_idx].name} | Success: {1 if reward > 0.9 else 0}")

    total_time = time.time() - start_time
    summary = logger.get_summary()
    print(f"\n>>> {name} Complete in {total_time:.1f}s")
    print(f"    Summary: {summary}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    
    # Run Baseline
    baseline = run_training("Baseline", use_lm=False, max_episodes=args.episodes)
    
    # Run LM-Enabled
    lm_enabled = run_training("LM-Enabled", use_lm=True, max_episodes=args.episodes)
    
    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"Baseline Phase:   {baseline.get('last_phase')}")
    print(f"LM-Enabled Phase: {lm_enabled.get('last_phase')}")
    print(f"Baseline Success: {baseline.get('success_rate_recent', 0):.2%}")
    print(f"LM-Enabled Success: {lm_enabled.get('success_rate_recent', 0):.2%}")
    
    # Success criterion: LM-Enabled reaches a later phase or has higher success rate
    # Phases are ordered by phase value in CurriculumManager
    print("\nRESULT: ", end="")
    if lm_enabled.get('success_rate_recent', 0) > baseline.get('success_rate_recent', 0) + 0.05:
        print("SUCCESS - LM version outperformed baseline.")
    else:
        print("INCONCLUSIVE - Baseline matched or exceeded LM version.")
